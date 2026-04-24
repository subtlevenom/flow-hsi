import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List

# Предполагается наличие этих метрик в вашем проекте
from ..metrics import PSNR, SSIM, DeltaE


class LogCoshLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true + 1e-12)))


class HSGAPipeline_v12(L.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 optimizer: str = 'adamw',
                 lr: float = 1e-3,
                 warmup_epochs: int = 15,
                 weight_decay: float = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Лоссы v12
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.logcosh_loss = LogCoshLoss()

        # Метрики
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        """ Инициализация v12 USGS Optimized """
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    # Инициализация для Advanced_GFFN и Spectral блоков
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # Инициализация Hyper-FFN и проекций (делаем их "тихими" для старта)
                if any(x in name for x in
                       ['expert_heads', 'orchestrator_proj', 'chi_post']):
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, mean=0.0, std=0.0001)

                # Инициализация Channel-Mix в Xi-Net как Identity
                if 'xi_net.0' in name and isinstance(m, nn.Conv2d):
                    nn.init.eye_(m.weight.view(m.weight.size(0), -1))

            print(
                f'HGSA_v12 (USGS Optimized): Multi-Scale GFFN and Hyper-FFN initialized.'
            )

    def configure_optimizers(self):
        params_groups = {
            'backbone': [],  # Encoder2D_v12 + Xi-Net
            'experts': [],  # HyperFFN Heads
            'orchestra': [],  # Orchestrator MSAB
            'chi': [],  # Gated Chi-Net
            'aux': []  # usgs_to_img
        }

        for name, param in self.model.named_parameters():
            if 'xi_net' in name or 'encoder' in name:
                params_groups['backbone'].append(param)
            elif 'expert_heads' in name:
                params_groups['experts'].append(param)
            elif 'orchestrator' in name:
                params_groups['orchestra'].append(param)
            elif 'chi_' in name:
                params_groups['chi'].append(param)
            elif 'usgs_to_img' in name:
                params_groups['aux'].append(param)
            else:
                params_groups['backbone'].append(param)

        optimizer = optim.AdamW(
            [
                {
                    'params': params_groups['backbone'],
                    'lr': self.lr * 0.6
                },  # Чуть выше для GFFN
                {
                    'params': params_groups['experts'],
                    'lr': self.lr * 1.0,
                    'weight_decay': 1e-2
                },
                {
                    'params': params_groups['orchestra'],
                    'lr': self.lr * 0.8
                },
                {
                    'params': params_groups['chi'],
                    'lr': self.lr * 0.7
                },
                {
                    'params': params_groups['aux'],
                    'lr': self.lr
                }
            ],
            weight_decay=self.weight_decay)

        self.scheduler_switch_epoch = int(self.trainer.max_epochs * 0.7)

        # OneCycle для основной фазы
        self.scheduler_1 = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.scheduler_switch_epoch,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.2,
            div_factor=10,
            final_div_factor=50)

        # Экспоненциальное затухание для фазы DeltaE доводки
        self.scheduler_2 = optim.lr_scheduler.ExponentialLR(optimizer,
                                                            gamma=0.98)

        return {"optimizer": optimizer}

    def total_variation_loss(self, p_list):
        loss = 0
        for p in p_list:
            # p: [B, C, 5, H, W] -> (w, mu, sigma, gate, tau)
            diff_h = torch.abs(p[:, :, :, 1:, :] - p[:, :, :, :-1, :])
            diff_w = torch.abs(p[:, :, :, :, 1:] - p[:, :, :, :, :-1])
            loss += (diff_h.mean() + diff_w.mean())
        return loss / len(p_list) if p_list else 0

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if self.current_epoch < self.scheduler_switch_epoch:
            self.scheduler_1.step()
        else:
            if self.trainer.is_last_batch:
                self.scheduler_2.step()

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        main_out, usgs_out, p_list = self(src)

        main_out = torch.clamp(main_out, 0.0, 1.0)
        usgs_out = torch.clamp(usgs_out, 0.0, 1.0)

        # 1. Основной лосс (Log-Cosh для стабильности на Волге)
        loss_color = self.logcosh_loss(main_out, tgt)

        # 2. Структурный лосс
        ssim_val = self.ssim_metric(main_out, tgt)
        loss_ssim = 1.0 - torch.clamp(ssim_val, 0., 1.)

        # 3. Вспомогательные лоссы
        loss_aux = self.mae_loss(usgs_out, tgt)
        loss_tv = self.total_variation_loss(p_list)

        # 4. Финальная перцептивная фаза (DeltaE)
        if self.current_epoch >= self.scheduler_switch_epoch:
            loss_de = self.de_metric(main_out, tgt).mean() * 0.05
        else:
            loss_de = 0.0

        # Комбинированная формула v12
        loss = 1.0 * loss_color + 0.3 * loss_ssim + 0.1 * loss_aux + 0.02 * loss_tv + loss_de

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_de_boost', loss_de, prog_bar=False)

        with torch.no_grad():
            psnr_val = self.psnr_metric(main_out, tgt)
            self.log('train_psnr', psnr_val, prog_bar=True)
            self.log('train_ssim', ssim_val, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)
        y = torch.clamp(y, 0.0, 1.0)

        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)
        return de_val

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)
        y = torch.clamp(y, 0.0, 1.0)
        self.log('test_psnr', self.psnr_metric(y, tgt), prog_bar=True)
        self.log('test_de', self.de_metric(y, tgt).mean(), prog_bar=True)

    def forward(self, x):
        # Обертка для инференса
        return self.model(src=x)['res']
