import os
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List, Dict

# Предполагается, что метрики импортируются из вашего проекта
from ..metrics import (PSNR, SSIM, SAM, DeltaE)


class HSGAPipeline(L.LightningModule):

    def __init__(
            self,
            model: nn.Module,
            optimizer: str = 'adamw',
            lr: float = 2e-4,
            freeze_until_epoch: int = 20,
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
        # Метрики (инициализация предполагается внешней или через импорт)
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                # Стандартная инициализация
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # Инициализация ViT в Xi-Net (Xavier для внимания)
                if 'xi_net' in name and isinstance(m, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(m.in_proj_weight)
                    nn.init.constant_(m.in_proj_bias, 0)
                    nn.init.xavier_uniform_(m.out_proj.weight)

                # Хирургическая инициализация для голов параметров (v7: 5 параметров)
                if 'param_heads' in name and hasattr(m, 'weight'):
                    nn.init.normal_(m.weight, mean=0.0, std=0.0001)

            print('HGSA_v7: ViT-Xi and Dilated KAN Initialization Applied.')

    def configure_optimizers(self):
        # Группировка параметров для v7
        xi_params = []  # Xi-Net (ViT + Conv)
        encoder_params = []  # Encoder2D (cmKAN)
        param_heads_params = []  # USGS Heads
        chi_params = []  # Chi-Net (Dilated KAN)

        for name, param in self.model.named_parameters():
            if 'xi_net' in name:
                xi_params.append(param)
            elif 'encoder' in name:
                encoder_params.append(param)
            elif 'param_heads' in name:
                param_heads_params.append(param)
            elif 'chi_net' in name:
                chi_params.append(param)
            else:
                encoder_params.append(param)

        optimizer = optim.AdamW([
            {
                'params': xi_params,
                'lr': self.lr * 0.5,  # ViT учим осторожнее
                'weight_decay': self.weight_decay
            },
            {
                'params': encoder_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay
            },
            {
                'params': param_heads_params,
                'lr': self.lr * 0.5,
                'weight_decay': 5e-2  # Высокий WD для стабильности гауссиан
            },
            {
                'params': chi_params,
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
        print(
            f"HGSA_v7: Freezing 'param_heads' until epoch {self.freeze_until_epoch}"
        )
        for name, param in self.model.named_parameters():
            if 'param_heads' in name:
                param.requires_grad = False

    def on_train_epoch_start(self):
        if self.current_epoch == self.freeze_until_epoch:
            print(
                f"HGSA_v7: Unfreezing 'param_heads' at epoch {self.current_epoch}"
            )
            for name, param in self.model.named_parameters():
                if 'param_heads' in name:
                    param.requires_grad = True

    def on_before_optimizer_step(self, optimizer):
        # Клиппинг градиентов важен для KAN и ViT структур
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def total_variation_loss(self, model_output_params_list):
        loss = 0
        for p_map in model_output_params_list:
            # p_map: [B, C, 5, H, W]
            # Штрафуем mu(1), sigma(2), gate(3), tau(4)
            diff_h = torch.abs(p_map[:, :, 1:, 1:, :] -
                               p_map[:, :, 1:, :-1, :])
            diff_w = torch.abs(p_map[:, :, 1:, :, 1:] -
                               p_map[:, :, 1:, :, :-1])
            loss += (torch.mean(diff_h) + torch.mean(diff_w))
        return loss / len(model_output_params_list)

    def forward(self, x: torch.Tensor):
        # В v7 модель возвращает тензор напрямую
        return self.model(src=x)['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # Сбор параметров для TV-loss (теперь 5 параметров в v7)
        with torch.no_grad():
            feat = self.model.layers.hgsa.encoder(src)
            all_params = []
            for head in self.model.layers.hgsa.param_heads:
                p = head(feat)
                num_p = 5  # w, mu, sigma, gate, tau
                all_params.append(
                    p.view(src.shape[0], src.shape[1], num_p, *src.shape[2:]))

        y = self(src)

        loss_mae = self.mae_loss(y, tgt)
        loss_ssim = 1.0 - self.ssim_metric(y, tgt)

        loss_tv = torch.tensor(0.0, device=self.device)
        if self.current_epoch >= self.freeze_until_epoch:
            loss_tv = self.total_variation_loss(all_params)

        # Баланс лоссов для v7
        loss = 1.0 * loss_mae + 0.8 * loss_ssim + 0.05 * loss_tv

        self.log('train_loss', loss, prog_bar=True)
        self.log('loss_tv', loss_tv, prog_bar=False)

        with torch.no_grad():
            psnr_val = self.psnr_metric(y, tgt)
            self.log('train_psnr', psnr_val, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)

        mae_val = self.mae_loss(y, tgt)
        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('val_mae', mae_val, prog_bar=True, sync_dist=True)
        self.log('val_psnr', psnr_val, prog_bar=True, sync_dist=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)

        val_loss = 1.0 * mae_val + 0.8 * (1. - ssim_val)

        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)
        psnr_val = self.psnr_metric(y, tgt)
        self.log('test_psnr', psnr_val)
        return psnr_val

    def predict_step(self, batch, batch_idx):
        src, tgt = batch
        return self(src)
