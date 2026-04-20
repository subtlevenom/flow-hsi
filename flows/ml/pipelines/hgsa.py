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


# Предполагаем наличие Charbonnier или используем реализацию ниже
class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))


class HSGAPipeline_v3(L.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 optimizer: str = 'adamw',
                 lr: float = 2e-4,
                 weight_decay: float = 1e-4,
                 metrics_channels: List[int] = [0, 1, 2]) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.metrics_channels = metrics_channels

        self.mae_loss = nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                # 1. Инициализация сверток и линейных слоев (SiLU/ReLU)
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # 2. Специфическая инициализация для Attention (temperature)
                if 'temperature' in name:
                    nn.init.constant_(m, 1.0)

                # 3. Инициализация голов генератора Гауссиан (mu в центр, малая амплитуда)
                if 'param_head' in name and isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            print('Gaussian-USGS (cmKAN Encoder) initialization applied.')

    def configure_optimizers(self):
        # Разделяем параметры на 3 группы для стабильности тяжелого энкодера
        encoder_params = []  # DWT, Transformer, Attention
        hypernet_params = []  # param_head (генерация mu, sigma, w)
        projector_params = []  # xi_proj, chi, lambdas

        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'param_head' in name:
                hypernet_params.append(param)
            else:
                projector_params.append(param)

        optimizer = optim.AdamW([{
            'params': encoder_params,
            'lr': self.lr,
            'weight_decay': self.weight_decay
        }, {
            'params': projector_params,
            'lr': self.lr,
            'weight_decay': self.weight_decay
        }, {
            'params': hypernet_params,
            'lr': self.lr * 0.5,
            'weight_decay': self.weight_decay * 0.1
        }])

        # Используем CosineAnnealing с теплым стартом (warmup) для Attention слоев
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def safe_de_loss(self, y, tgt):
        de = self.de_metric(y, tgt)
        de = torch.where(torch.isnan(de), torch.zeros_like(de), de)
        # Clamping важен для Гауссиан, чтобы избежать бесконечных градиентов в начале
        return torch.clamp(de, min=1e-3, max=50.0).mean()

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # В v2 модель возвращает тензор напрямую
        return self.model(src=x, tgt=y)['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        y = self.forward(src)

        # 1. Charbonnier для PSNR
        loss_charb = self.charbonnier_loss(y, tgt)
        # 2. SSIM для структурности (cmKAN силен в этом)
        ssim_loss = self.ssim_metric(y, tgt)
        loss_ssim = 1.0 - ssim_loss

        # Итоговый комбинированный лосс
        loss = 1.0 * loss_charb + 0.5 * loss_ssim

        with torch.no_grad():
            l1_val = self.mae_loss(y, tgt)
            psnr_val = self.psnr_metric(y, tgt)
            de_val = self.safe_de_loss(y, tgt)
            self.log('mae', l1_val, prog_bar=True, logger=True)
            self.log('psnr', psnr_val, prog_bar=True, logger=True)
            self.log('ssim', ssim_loss, prog_bar=True, logger=True)
            self.log('de', de_val, prog_bar=True, logger=True)
            self.log('train_loss', loss, prog_bar=True, logger=True)

        if torch.isnan(loss):
            print("Warning: NaN loss detected! Skipping step.")
            return None

        return loss

    def on_before_optimizer_step(self, optimizer):
        # Гауссианы чувствительны к резким скачкам, клиппинг обязателен
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self.forward(src)

        psnr_val = self.psnr_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()
        ssim_val = self.ssim_metric(y, tgt)

        # Теперь главная метрика для сохранения чекпоинтов - PSNR
        self.log('val_psnr', psnr_val, prog_bar=True, sync_dist=True)
        self.log('val_de', de_val, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)

        # val_loss теперь ориентирован на качество, а не только на цвет
        combined_val = (1.0 - ssim_val) + (de_val / 20.0)
        self.log('val_loss', combined_val, prog_bar=True)

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
            print(
                f"[{name[0]}] PSNR: {psnr:.2f} | AVG dE: {self.sum_de/n:.2f}")

        return y
