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
from flows.tools.utils import text
from flows.tools.utils import models
from flows.core import Logger
from flows.ml.losses import GPDFLoss
from ..models import Flow
from ..metrics import (PSNR, SSIM, SAM, DeltaE)
from flows.ml.layers.sep_gpd import MultivariateNormal


class GGPDPipeline(L.LightningModule):

    def __init__(self,
                 model: Flow,
                 optimizer: str = 'adam',
                 lr: float = 1e-3,
                 weight_decay: float = 0,
                 warmup_epochs: int = 0,
                 hgsa_ckpt: str = None,
                 sam_weight: float = 0.1,
                 metrics_channels: List[int] = [0, 1, 2]) -> None:
        super(GGPDPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        # Warm-up: freeze the pretrained HGSA color-transport core for the
        # first ``warmup_epochs`` epochs so the surrounding MSAB encoder/
        # decoder can adapt to it before the whole network is fine-tuned.
        self.warmup_epochs = warmup_epochs
        self.hgsa_ckpt = hgsa_ckpt
        # Spectral-angle (SAM) loss weight — key spectral-fidelity term for
        # the HSI task; MAE alone matches per-band intensity but ignores
        # the shape of the spectral signature.
        self.sam_weight = sam_weight
        self._hgsa_frozen = False
        self.ggpd_loss = GPDFLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.de_metric = DeltaE()
        self.sam_metric = SAM()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.metrics_channels = metrics_channels

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        '''
        Initialize model weights, load the pretrained HGSA color-transport
        core, and (optionally) freeze it for the warm-up epochs so the
        surrounding MSAB encoder/decoder can adapt to the frozen core
        before the whole network is fine-tuned jointly.
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

            # Load the pretrained HGSA core over the freshly-initialized
            # weights, then freeze it for the warm-up phase. Only done for
            # training — for test/predict the full checkpoint is restored
            # by Lightning and must not be partially overwritten here.
            if self.hgsa_ckpt and os.path.isfile(self.hgsa_ckpt):
                models.load_model(
                    self.model.layers.hgsa, '_model', self.hgsa_ckpt)
                Logger.info(
                    f'Loaded pretrained HGSA core from {self.hgsa_ckpt}.')
            else:
                Logger.info(
                    'No HGSA checkpoint found; training HGSA from scratch.')

            if self.warmup_epochs > 0:
                self._set_hgsa_frozen(True)
                Logger.info(
                    f'HGSA core frozen for the first {self.warmup_epochs} '
                    f'warm-up epoch(s).')

            Logger.info('Initialized model weights with isp pipeline.')

    def _set_hgsa_frozen(self, frozen: bool) -> None:
        '''Freeze/unfreeze the pretrained HGSA color-transport core.'''
        for p in self.model.layers.hgsa.parameters():
            p.requires_grad = not frozen
        self._hgsa_frozen = frozen

    def on_train_epoch_start(self) -> None:
        # Unfreeze the HGSA core once the warm-up phase is over so the
        # whole network is fine-tuned jointly.
        if self._hgsa_frozen and self.current_epoch >= self.warmup_epochs:
            self._set_hgsa_frozen(False)
            Logger.info(f'Unfroze HGSA core at epoch {self.current_epoch}.')

    def spectral_angle_loss(self, pred: torch.Tensor,
                            tgt: torch.Tensor) -> torch.Tensor:
        '''Mean spectral angle (radians) across the spectral dimension.

        Central spectral-fidelity objective for the HSI task. Fully
        differentiable and guarded against the 0/0 (zero-spectrum) and
        arccos(±1) singularities so gradients stay finite.
        '''
        p = pred.flatten(2)                       # [B, C, H*W]
        t = tgt.flatten(2)
        dot = (p * t).sum(dim=1)                   # [B, H*W]
        denom = p.norm(dim=1) * t.norm(dim=1) + 1e-8
        cos = (dot / denom).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.arccos(cos).mean()

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.weight_decay)
        else:
            raise ValueError(
                f'unsupported optimizer_type: {self.optimizer_type}')
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=1, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pred = self.model(src=x, tgt=y)
        return pred['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src, tgt).to(torch.float32)

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        sam_ang = self.spectral_angle_loss(y, tgt)
        # de_loss = self.de_metric(y[:, self.metrics_channels], tgt[:, self.metrics_channels])
        loss = mae_loss + self.sam_weight * sam_ang

        self.log('mae', mae_loss, prog_bar=True, logger=True)
        self.log('psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('sam', sam_loss, prog_bar=True, logger=True)
        # self.log('de', de_loss, prog_bar=True, logger=True)
        self.log('hgsa_frozen', float(self._hgsa_frozen),
                 prog_bar=True, logger=True)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src, tgt).to(torch.float32)

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        # de_loss = self.de_metric(y[:, self.metrics_channels], tgt[:, self.metrics_channels])
        loss = mae_loss + self.sam_weight * sam_loss

        self.log('val_mae', mae_loss, prog_bar=True, logger=True)
        self.log('val_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('val_sam', sam_loss, prog_bar=True, logger=True)
        # self.log('val_de', de_loss, prog_bar=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src, tgt).to(torch.float32)

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        # de_loss = self.de_metric(y[:, self.metrics_channels], tgt[:, self.metrics_channels])
        loss = mae_loss + self.sam_weight * sam_loss

        self.log('test_mae', mae_loss, prog_bar=True, logger=True)
        self.log('test_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('test_sam', sam_loss, prog_bar=True, logger=True)
        # self.log('test_de', de_loss, prog_bar=True, logger=True)
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
        y = self(src, tgt)
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
