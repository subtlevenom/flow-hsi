import os
import random
import statistics
from typing import List
from einops import rearrange
import torch
from torch import nn
import lightning as L
from torch import optim
import torch.nn.functional as F
import torchvision
import time
from flows.tools.utils import models, text
from flows.core import Logger
from flows.ml.losses import GPDFLoss
from ..models import Flow
from ..metrics import (PSNR, SSIM, SAM, DeltaE)
from flows.ml.layers.sep_gpd import MultivariateNormal


class USGSPipeline(L.LightningModule):

    def __init__(self,
                 model: Flow,
                 optimizer: str = 'adam',
                 lr: float = 1e-3,
                 weight_decay: float = 0,
                 lambda_reg: float = 1e-5,
                 metrics_channels: List[int] = [0, 1, 2]) -> None:
        super(USGSPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.ggpd_loss = GPDFLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.de_metric = DeltaE()
        self.sam_metric = SAM()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.metrics_channels = metrics_channels
        self.lambda_reg = lambda_reg

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        '''
        Initialize model weights before training
        '''
        if stage == 'fit' or stage is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # MODEL_PATH = '.experiments/ggpd.huawei/logs/checkpoints/_last.ckpt'
        # models.load_model(self.model.layers.encoder, 'model.layers.encoder', MODEL_PATH)
        # models.require_grad(self.model.layers.encoder.gpd_x, requires_grad=False)

        Logger.info('Initialized model weights with isp pipeline.')

    def configure_optimizers(self):
        if self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,
            )
        elif self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=500,
                T_mult=1,
                eta_min=1e-5,
            )
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=500,
                T_mult=1,
                eta_min=1e-5,
            )
        else:
            raise ValueError(
                f'unsupported optimizer_type: {self.optimizer_type}')
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def laplacian_pyramid_loss(self, img1, img2, levels=3):
        """Лосс на основе пирамид Лапласа для сохранения деталей."""

        def build_pyramid(img, levels):
            pyramid = []
            current = img
            for _ in range(levels):
                down = F.interpolate(current,
                                     scale_factor=0.5,
                                     mode='bilinear',
                                     align_corners=False)
                up = F.interpolate(down,
                                   size=current.shape[-2:],
                                   mode='bilinear',
                                   align_corners=False)
                pyramid.append(current - up)
                current = down
            pyramid.append(current)
            return pyramid

        pyr1 = build_pyramid(img1, levels)
        pyr2 = build_pyramid(img2, levels)
        return sum(F.l1_loss(a, b) for a, b in zip(pyr1, pyr2))

    def gaussian_regularization(self, a, mu, sigma):
        """
        Регуляризация согласно Теореме 2 (Tikhonov regularization) 
        и предотвращение вырождения сигмы.
        """
        # 1. Штраф за слишком большие амплитуды (стабильность)
        reg_a = torch.norm(a, p=2)

        # 2. Штраф за выход центров за пределы [-1, 1]
        reg_mu = torch.mean(F.relu(torch.abs(mu) - 1.0))

        # 3. Штраф за слишком узкие Гауссианы (Nyquist criterion)
        # Если сигма меньше шага дискретизации, возникнет алиасинг.
        step = 2.0 / self.model.layers.usgs.num_nodes
        reg_sigma = torch.mean(F.relu(step - sigma))

        return reg_a + reg_mu + reg_sigma

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pred = self.model(src=x, tgt=y)
        return pred['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # Финальный результат модели
        y, a, mu, sigma = self.model.layers.usgs(src, True)

        # Основные лоссы
        l1 = self.mae_loss(y, tgt)
        laplace = self.laplacian_pyramid_loss(y, tgt)

        # Регуляризация
        reg = self.gaussian_regularization(a, mu, sigma)

        total_loss = l1 + 0.5 * laplace + self.lambda_reg * reg

        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        de_loss = self.de_metric(y[:, self.metrics_channels],
                                 tgt[:, self.metrics_channels])
        loss = total_loss

        self.log('mae', total_loss, prog_bar=True, logger=True)
        self.log('psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('sam', sam_loss, prog_bar=True, logger=True)
        self.log('de', de_loss, prog_bar=True, logger=True)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src)

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        de_loss = self.de_metric(y[:, self.metrics_channels],
                                 tgt[:, self.metrics_channels])
        loss = mae_loss

        self.log('val_mae', mae_loss, prog_bar=True, logger=True)
        self.log('val_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('val_sam', sam_loss, prog_bar=True, logger=True)
        self.log('val_de', de_loss, prog_bar=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src)

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        de_loss = self.de_metric(y[:, self.metrics_channels],
                                 tgt[:, self.metrics_channels])
        loss = mae_loss

        self.log('test_mae', mae_loss, prog_bar=True, logger=True)
        self.log('test_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('test_sam', sam_loss, prog_bar=True, logger=True)
        self.log('test_de', de_loss, prog_bar=True, logger=True)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    sum_mae = 0
    sum_psnr = 0
    sum_ssim = 0
    sum_sam = 0
    sum_de = 0
    start_time = 0

    def predict_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.start_time = time.perf_counter()

        src, tgt, name = batch
        y = self(src)
        elapsed = time.perf_counter() - self.start_time

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        de_loss = self.de_metric(y[:, self.metrics_channels],
                                 tgt[:, self.metrics_channels])

        self.sum_mae += mae_loss
        self.sum_psnr += psnr_loss
        self.sum_ssim += ssim_loss
        self.sum_sam += sam_loss
        self.sum_de += de_loss
        n = 1 + batch_idx

        text.print_json({
            name[0]: {
                'CUR': {
                    'mae': mae_loss.item(),
                    'psnr': psnr_loss.item(),
                    'ssim': ssim_loss.item(),
                    'sam': sam_loss.item(),
                    'de': de_loss.item(),
                },
                'AVG': {
                    'mae': self.sum_mae.item() / n,
                    'psnr': self.sum_psnr.item() / n,
                    'ssim': self.sum_ssim.item() / n,
                    'sam': self.sum_sam.item() / n,
                    'de': self.sum_de.item() / n,
                },
                'TIME': elapsed / n,
            },
        })

        return {'loss': de_loss}
