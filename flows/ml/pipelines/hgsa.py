import os
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List, Dict

# Предполагаем, что метрики импортируются из вашего окружения
from ..metrics import (PSNR, SSIM, SAM, DeltaE)
from flows.core import Logger


class HSGAPipeline_v2(L.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 optimizer: str = 'adam',
                 lr: float = 2e-4, 
                 weight_decay: float = 1e-5,
                 metrics_channels: List[int] = [0, 1, 2]) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.metrics_channels = metrics_channels

        self.mae_loss = nn.L1Loss()
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()
        
        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    # Стандартная инициализация Kaiming
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # 1. Инициализация Гауссова ядра (делаем амплитуды маленькими в начале)
                if 'gaussian_core.hyper' in name:
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.constant_(m.weight, 0)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # 2. Матричная голова: инициализируем так, чтобы в начале был Identity
                if 'matrix_head' in name:
                    if isinstance(m, nn.Conv2d) and m.out_channels == 12:
                        nn.init.constant_(m.weight, 0)
                        nn.init.constant_(m.bias, 0)

            print('HGSA-v2 Identity initialization applied.')

    def configure_optimizers(self):
        # Выделяем параметры Гауссова ядра в отдельную группу (учим чуть осторожнее)
        gaussian_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if 'gaussian_core' in name:
                gaussian_params.append(param)
            else:
                base_params.append(param)

        optimizer = optim.AdamW(
            [
                {'params': base_params, 'lr': self.lr},
                {'params': gaussian_params, 'lr': self.lr * 0.5} # Не 0.1, так как ядро теперь плоское
            ],
            weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # В v2 модель возвращает тензор напрямую
        return self.model(src=x, tgt=y)['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        y = self.forward(src)

        l1_loss = self.mae_loss(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)

        # Комбинированный лосс для PSNR и структурной целостности
        loss = 1.0 * l1_loss + 0.5 * (1.0 - ssim_loss)

        with torch.no_grad():
            psnr_val = self.psnr_metric(y, tgt)
            de_val = self.de_metric(y, tgt)
            self.log('mae', l1_loss, prog_bar=True, logger=True)
            self.log('psnr', psnr_val, prog_bar=True, logger=True)
            self.log('ssim', ssim_loss, prog_bar=True, logger=True)
            self.log('de', de_val, prog_bar=True, logger=True)
            self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        # Клиппинг важен для предотвращения взрывов в экспонентах Гауссиан
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self.forward(src)

        psnr_val = self.psnr_metric(y, tgt)
        de_val = self.de_metric(y, tgt)

        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)
        self.log('val_loss', de_val, prog_bar=True)

        return de_val

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)
        psnr_val = self.psnr_metric(y, tgt)
        de_val = self.de_metric(y[:, self.metrics_channels],
                                tgt[:, self.metrics_channels])
        self.log('test_psnr', psnr_val)
        self.log('test_de', de_val)
        self.log('val_loss', de_val, prog_bar=True)

        return de_val

    def predict_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.reset_predict_metrics()
            self.start_time = time.perf_counter()

        src, tgt, name = batch
        y = self(src)
        elapsed = time.perf_counter() - self.start_time

        mae = self.mae_loss(y, tgt).item()
        psnr = self.psnr_metric(y, tgt).item()
        de = self.de_metric(y[:, self.metrics_channels],
                            tgt[:, self.metrics_channels]).item()

        self.sum_mae += mae
        self.sum_psnr += psnr
        self.sum_de += de
        n = batch_idx + 1

        if batch_idx % 5 == 0:
            print(f"[{name[0]}] PSNR: {psnr:.2f} | AVG dE: {self.sum_de/n:.2f}")

        return y