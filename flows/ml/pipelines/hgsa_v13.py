import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List

from ..metrics import PSNR, SSIM, DeltaE


class LogCoshLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        x = y_pred - y_true
        # Используем формулу: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        # Это предотвращает inf при больших x
        loss = torch.abs(x) + F.softplus(-2. * torch.abs(x)) - math.log(2.0)
        return torch.mean(loss)

class GradLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # torch.gradient возвращает (grad_y, grad_x) для 4D тензора по дим 2 и 3
        grad_y_pred, grad_x_pred = torch.gradient(pred, dim=(2, 3))
        grad_y_tgt, grad_x_tgt = torch.gradient(target, dim=(2, 3))
        
        loss_y = F.l1_loss(grad_y_pred, grad_y_tgt)
        loss_x = F.l1_loss(grad_x_pred, grad_x_tgt)
        
        return loss_y + loss_x


class HSGAPipeline_v13(L.LightningModule):

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
        self.grad_loss = GradLoss()

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
                    if any(x in name
                           for x in ['w_scales', 'mu_offsets', 'sigma_bases']):
                        continue  # Пропускаем
                    # Инициализация для Advanced_GFFN и Spectral блоков
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # Инициализация Hyper-FFN и проекций (делаем их "тихими" для старта)
                if any(x in name for x in
                       ['expert_heads', 'orchestrator.proj', 'chi_post']):
                    if isinstance(m, nn.Conv2d):
                        # nn.init.normal_(m.weight, mean=0.0, std=0.01)
                        nn.init.kaiming_normal_(m.weight,
                                                mode="fan_out",
                                                nonlinearity="relu")

                # Инициализация Channel-Mix в Xi-Net как Identity
                if 'xi_net.1' in name and isinstance(m, nn.Conv2d):
                    nn.init.eye_(m.weight.view(m.weight.size(0), -1))

            print(
                f'HGSA_v12 (USGS Optimized): Multi-Scale GFFN and Hyper-FFN initialized.'
            )

    def configure_optimizers(self):
        params_groups = {
            'backbone': [],
            'experts': [],
            'orchestra': [],
            'chi': [],
            'aux': [],  # usgs_to_img
            'global_params':
            [],  # Новая группа для w_scales, mu_offsets, sigma_bases
        }

        for name, param in self.model.named_parameters():
            if any(x in name
                   for x in ['w_scales', 'mu_offsets', 'sigma_bases']):
                params_groups['global_params'].append(param)
            elif 'xi_net' in name or 'encoder' in name:
                params_groups['backbone'].append(param)
            elif 'expert_heads' in name:
                params_groups['experts'].append(param)
            elif 'orchestrator' in name:
                params_groups['orchestra'].append(param)
            elif 'chi_net' in name:
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
                },
                {
                    'params': params_groups['global_params'],
                    'lr': self.lr * 0.5,
                    'weight_decay': 0.0
                },
            ],
            weight_decay=self.weight_decay)

        # Сначала количество эпох до переключения
        self.scheduler_switch_epoch = int(self.trainer.max_epochs * 0.7)
        # Сначала считаем, сколько всего шагов в одной эпохе
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        # Считаем шаги только для первой фазы (OneCycle)
        first_phase_steps = self.scheduler_switch_epoch * steps_per_epoch

        # OneCycle для основной фазы
        self.scheduler_1 = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.scheduler_switch_epoch,
            total_steps=first_phase_steps,
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

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        if self.current_epoch < self.scheduler_switch_epoch:
            self.scheduler_1.step()
        else:
            if self.trainer.is_last_batch:
                self.scheduler_2.step()

    def on_before_optimizer_step(self, optimizer):
        # MSAB может давать всплески градиента, клиппинг обязателен
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        main_out, usgs_out, p_list = self(src)

        main_out_c = torch.clamp(main_out, 0.0, 1.0)

        # 1. Основной лосс (Log-Cosh для стабильности на Волге)
        loss_color = self.logcosh_loss(main_out, tgt)
        loss_grad = self.grad_loss(main_out, tgt)

        # 2. Структурный лосс
        ssim_val = self.ssim_metric(main_out_c, tgt)
        loss_ssim = 1.0 - torch.clamp(ssim_val, 0., 1.)

        # 3. Вспомогательные лоссы
        loss_aux = self.mae_loss(usgs_out, tgt)
        loss_tv = self.total_variation_loss(p_list)

        # 4. Финальная перцептивная фаза (DeltaE)
        if self.current_epoch < self.scheduler_switch_epoch:
            loss_de = 0.0
        else:
            loss_de = self.de_metric(main_out_c, tgt).mean() * 0.05

        # 5. Warmup
        if self.current_epoch < self.warmup_epochs:
            a, b = 1.0, 0.1
        else:
            a, b = 1.0, 0.1

        # 6. Комбинированная формула v12
        loss = a * loss_grad + a * loss_color + 0.3 * loss_ssim + b * loss_aux + 0.02 * loss_tv + loss_de

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_de_boost', loss_de, prog_bar=False)

        with torch.no_grad():
            psnr_val = self.psnr_metric(main_out_c, tgt)
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
        self.log('val_loss', de_val, prog_bar=False)

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
