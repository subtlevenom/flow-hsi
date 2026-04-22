import os
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List, Dict

# Предполагается, что метрики импортируются из вашего проекта
from ..metrics import (PSNR, SSIM, SAM, DeltaE)


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
            freeze_until_epoch: int = 30,  # Уменьшил порог, так как v5 сложнее
            weight_decay: float = 1e-4,
            metrics_channels: List[int] = [0, 1, 2]) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.metrics_channels = metrics_channels
        self.freeze_until_epoch = freeze_until_epoch

        # Лоссы
        self.charbonnier_loss = CharbonnierLoss()
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                # Стандартная инициализация сверток
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # Хирургическая инициализация для голов параметров Гауссиан (v5: param_heads)
                # Мы инициализируем их очень малыми значениями, чтобы mu и sigma 
                # в начале обучения определялись только bias-ами или средними значениями активаций.
                if 'param_heads' in name and hasattr(m, 'weight'):
                    nn.init.normal_(m.weight, mean=0.0, std=0.0001)

            print('HGSA_v5: Channel-wise FFN Heads Initialization Applied.')

    def configure_optimizers(self):
        # Группировка параметров для v5
        encoder_params = []      # Основное тело (cmKAN)
        param_heads_params = []  # Список FFN блоков для каждой гауссианы
        other_params = []        # xi_proj, chi_net

        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'param_heads' in name:
                param_heads_params.append(param)
            else:
                other_params.append(param)

        optimizer = optim.AdamW([
            {
                'params': encoder_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay
            },
            {
                'params': param_heads_params,
                'lr': self.lr * 0.5,  # В v5 головы сложнее, даем чуть больше LR чем в v4
                'weight_decay': 5e-2   # Повышенный WD для предотвращения резких пиков в mu/sigma
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

    def on_train_start(self):
        print(f"HGSA_v5: Freezing 'param_heads' until epoch {self.freeze_until_epoch}")
        for name, param in self.model.named_parameters():
            if 'param_heads' in name:
                param.requires_grad = False

    def on_train_epoch_start(self):
        if self.current_epoch == self.freeze_until_epoch:
            print(f"HGSA_v5: Unfreezing 'param_heads' at epoch {self.current_epoch}")
            for name, param in self.model.named_parameters():
                if 'param_heads' in name:
                    param.requires_grad = True

    def on_before_optimizer_step(self, optimizer):
        # Клиппинг градиентов для стабильности USGS
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def total_variation_loss(self, model_output_params_list):
        """
        TV-loss для списка параметров. 
        params_list содержит тензоры [B, C*3, H, W] для каждой гауссианы.
        """
        loss = 0
        for p_map in model_output_params_list:
            # Штрафуем только mu и sigma (каналы 1 и 2 в размерности 3)
            # p_map: [B, C, 3, H, W]
            diff_h = torch.abs(p_map[:, :, 1:, 1:, :] - p_map[:, :, 1:, :-1, :])
            diff_w = torch.abs(p_map[:, :, 1:, :, 1:] - p_map[:, :, 1:, :, :-1])
            loss += (torch.mean(diff_h) + torch.mean(diff_w))
        return loss / len(model_output_params_list)

    def forward(self, x: torch.Tensor):
        # В v5 модель принимает только x
        return self.model(src=x)['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # Для TV-loss в v5 нам нужно собрать выходы всех param_heads
        # Это можно сделать, прогнав encoder один раз
        with torch.set_grad_enabled(True):
            feat = self.model.layers.hgsa.encoder(src)
            all_params = []
            for head in self.model.layers.hgsa.param_heads:
                p = head(feat)
                all_params.append(p.view(src.shape[0], src.shape[1], 3, *src.shape[2:]))

        y = self(src)

        # 1. Основные лоссы
        loss_charb = self.charbonnier_loss(y, tgt)
        loss_ssim = 1.0 - self.ssim_metric(y, tgt)

        # 2. TV Loss (контроль пространственной гладкости mu и sigma)
        loss_tv = torch.tensor(0.0, device=self.device)
        if self.current_epoch >= self.freeze_until_epoch:
            loss_tv = self.total_variation_loss(all_params)

        # Итоговый баланс для v5
        # Увеличил вес SSIM, так как конкатенация в Chi-net может давать шум
        loss = 1.0 * loss_charb + 0.7 * loss_ssim + 0.1 * loss_tv

        self.log('train_loss', loss, prog_bar=True)
        self.log('loss_tv', loss_tv, prog_bar=False)
        
        with torch.no_grad():
            psnr_val = self.psnr_metric(y, tgt)
            self.log('train_psnr', psnr_val, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)

        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('val_psnr', psnr_val, prog_bar=True, sync_dist=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)

        # Комбинированный лосс для выбора лучшего чекпоинта
        val_loss = (1.0 - ssim_val) + (de_val / 40.0)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)
        psnr_val = self.psnr_metric(y, tgt)
        self.log('test_psnr', psnr_val)
        return psnr_val

    def predict_step(self, batch, batch_idx):
        src, tgt, name = batch
        y = self(src)
        return y