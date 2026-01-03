import os
import torch
from torch import nn
import lightning as L
from torch import optim
import torch.nn.functional as F
import torchvision
from ..models import Flow
from flows.core import Logger
from ..metrics import (PSNR, SSIM, DeltaE)
from tools.utils import models
import time


class DefaultPipeline(L.LightningModule):

    def __init__(
        self,
        model: Flow,
        optimizer: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 0,
    ) -> None:
        super(DefaultPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.de_metric = DeltaE()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))

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
                    nn.init.constant_(m.bias, 0)

        # MODEL_PATH = '.experiments/hsgaussian.weighted.7.cave-hsi.v8/logs/checkpoints/__last.ckpt'
        # MODEL_PATH = '/data/korepanov/models/cmkan.weighted.cave.v8/logs/checkpoints/last.ckpt'
        # models.load_model(self.model.layers, 'model.layers', MODEL_PATH)
        # models.load_model(self.model.layers.encoder, 'model.layers.encoder', MODEL_PATH)
        # models.load_model(self.model.layers.decoder, 'model.layers.decoder', MODEL_PATH)
        # models.require_grad(self.model.layers.encoder, requires_grad=False)
        # models.require_grad(self.model.layers.decoder, requires_grad=False)

        Logger.info('Initialized model weights with isp pipeline.')

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

    def forward(self, x: torch.Tensor, scale:int=0) -> torch.Tensor:
        pred = self.model(image=x)
        return pred['result']

    def training_step(self, batch, batch_idx):
        src, target = batch
        prediction = self(src)

        mae_loss = self.mae_loss(prediction, target)
        psnr_loss = self.psnr_metric(prediction, target)
        ssim_loss = self.ssim_metric(prediction, target)
        loss = mae_loss + 0.15 * (1.-ssim_loss) #+ 0.2 * (40. - psnr_loss)

        self.log('train_mae', mae_loss, prog_bar=True, logger=True)
        self.log('train_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('train_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        src, target = batch
        prediction = self(src)

        mae_loss = self.mae_loss(prediction, target)
        psnr_loss = self.psnr_metric(prediction, target)
        ssim_loss = self.ssim_metric(prediction, target)
        de_loss = self.de_metric(prediction[:,[5,15,25]], target[:,[5,15,25]])

        self.log('val_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('val_de', de_loss, prog_bar=True, logger=True)
        self.log('val_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}

    def test_step(self, batch, batch_idx):
        src, target = batch
        prediction = self(src)

        mae_loss = self.mae_loss(prediction, target)
        psnr_loss = self.psnr_metric(prediction, target)
        ssim_loss = self.ssim_metric(prediction, target)
        de_loss = self.de_metric(prediction[:,[5,15,25]], target[:,[5,15,25]])

        self.log('test_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('test_loss', de_loss, prog_bar=True, logger=True)
        return {'loss': de_loss}

    sum = 0
    start_time = 0

    def predict_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.start_time = time.perf_counter()

        src, target, name = batch
        prediction = self(src)
        elapsed = time.perf_counter() - self.start_time 

        mae_loss = self.mae_loss(prediction, target)
        psnr_loss = self.psnr_metric(prediction, target)
        ssim_loss = self.ssim_metric(prediction, target)
        de_loss = self.de_metric(prediction[:,[5,15,25]], target[:,[5,15,25]])

        self.sum += psnr_loss

        print(f'{name[0]}: psnr {psnr_loss}, ssim {ssim_loss}, loss {de_loss} | AVG: {self.sum / (1 + batch_idx)} | Elapsed: {elapsed/(batch_idx + 1)}')

        return {'loss': de_loss}
