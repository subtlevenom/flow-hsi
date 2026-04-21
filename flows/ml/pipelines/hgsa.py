import os
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List, Dict

# Метрики и логгеры
from ..metrics import (PSNR, SSIM, SAM, DeltaE)
from flows.core import Logger


class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))


class HSGAPipeline(L.LightningModule):

    def __init__(
            self,
            model: nn.Module,
            optimizer: str = 'adamw',
            lr: float = 2e-4,
            freeze_until_epoch: int = 50,  # Порог разморозки
            weight_decay: float = 1e-4,
            metrics_channels: List[int] = [0, 1, 2]) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.metrics_channels = metrics_channels
        self.freeze_until_epoch = freeze_until_epoch

        # Лоссы
        self.mae_loss = nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # Инициализация головы параметров (очень малые веса для стабильного старта)
                if 'param_head' in name and hasattr(m, 'weight'):
                    nn.init.normal_(m.weight, mean=0.0, std=0.0001)

            print('HGSA_v4: Surgical Initialization Applied.')

    def configure_optimizers(self):
        # Группировка параметров для разного LR и WD
        encoder_params = []  # Основное тело (cmKAN)
        param_head_params = []  # Генератор Гауссиан (Источник хаоса)
        other_params = []  # xi_proj, chi, lambdas

        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'param_head' in name:
                param_head_params.append(param)
            else:
                other_params.append(param)

        optimizer = optim.AdamW([
            {
                'params': encoder_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay
            },
            {
                'params': param_head_params,
                'lr': self.lr * 0.25,  # Замедляем обучение головы параметров
                'weight_decay':
                1e-2  # Сильный Weight Decay для сглаживания параметров
            },
            {
                'params': other_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay
            }
        ])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def on_before_optimizer_step(self, optimizer):
        # Клиппинг 1.0 для стабильности KAN-слоев
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def on_train_start(self):
        # В начале обучения замораживаем генератор параметров Гауссиан
        print(
            f"HGSA_v4: Freezing 'param_head' until epoch {self.freeze_until_epoch}"
        )
        for name, param in self.model.named_parameters():
            if 'param_head' in name:
                param.requires_grad = False

    def on_train_epoch_start(self):
        # Автоматическая разморозка при достижении нужной эпохи
        if self.current_epoch == self.freeze_until_epoch:
            print(
                f"HGSA_v4: Unfreezing 'param_head' at epoch {self.current_epoch}"
            )
            for name, param in self.model.named_parameters():
                if 'param_head' in name:
                    param.requires_grad = True

            # Пересобираем оптимизатор, чтобы включить новые параметры
            # (Lightning подхватит изменения, если вернуть новый оптимизатор в configure_optimizers)
            # Но проще сразу создать оптимизатор со всеми параметрами,
            # так как requires_grad=False просто исключит их из расчетов.

    def on_before_optimizer_step(self, optimizer):
        # Увеличиваем порог клиппинга до 1.0, чтобы не душить полезные градиенты энкодера,
        # но при этом сглаживать выбросы USGS.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def total_variation_loss(self, params_map):
        """Штраф за резкие скачки параметров между соседними пикселями (убивает высокий DE)"""
        diff_h = torch.abs(params_map[:, :, 1:, :] - params_map[:, :, :-1, :])
        diff_w = torch.abs(params_map[:, :, :, 1:] - params_map[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        return self.model(src=x, tgt=y)['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # Получаем параметры для TV-loss (нужно для контроля хаоса)
        # В HGSA_v3 feat = self.encoder(x), params = self.param_head(feat)
        feat = self.model.layers.hgsa.encoder(src)
        params = self.model.layers.hgsa.param_head(feat)

        y = self(src)

        # 1. Основные лоссы
        loss_charb = self.charbonnier_loss(y, tgt)
        loss_ssim = 1.0 - self.ssim_metric(y, tgt)

        # 2. TV Loss включается только после разморозки
        loss_tv = torch.tensor(0.0, device=self.device)
        if self.current_epoch >= self.freeze_until_epoch:
            loss_tv = self.total_variation_loss(params)

        # Итоговый баланс:
        # До 50 эпохи: только цвет и структура.
        # После 50: добавляется штраф за "дребезг" параметров.
        loss = 1.0 * loss_charb + 0.5 * loss_ssim + 0.1 * loss_tv

        with torch.no_grad():
            psnr_val = self.psnr_metric(y, tgt)
            de_val = self.de_metric(y, tgt).mean()
            self.log('train_psnr', psnr_val, prog_bar=True)
            self.log('train_de', de_val, prog_bar=True)
            self.log('loss_tv', loss_tv, prog_bar=(loss_tv > 0))
            self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src)

        psnr_val = self.psnr_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()
        ssim_val = self.ssim_metric(y, tgt)

        self.log('val_psnr', psnr_val, prog_bar=True, sync_dist=True)
        self.log('val_de', de_val, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)

        # Целевая метрика для чекпоинтов: баланс чистоты (DE) и точности (PSNR)
        # Мы хотим минимизировать это значение
        combined_val = (1.0 - ssim_val) + (de_val / 30.0)
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
