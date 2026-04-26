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
        # log(cosh(x)) аппроксимация для стабильности
        loss = torch.abs(x) + F.softplus(-2. * torch.abs(x)) - math.log(2.0)
        return torch.mean(loss)


class GradLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Вычисляем градиенты по H и W
        grads_pred = torch.gradient(pred, dim=(2, 3))
        grads_tgt = torch.gradient(target, dim=(2, 3))

        loss_y = F.l1_loss(grads_pred[0], grads_tgt[0])
        loss_x = F.l1_loss(grads_pred[1], grads_tgt[1])
        return loss_y + loss_x


class HSGAPipeline_v14(L.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 optimizer: str = 'adamw',
                 lr: float = 1e-3,
                 warmup_epochs: int = 10,
                 weight_decay: float = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Лоссы
        self.mae_loss = nn.L1Loss()
        self.logcosh_loss = LogCoshLoss()
        self.grad_loss = GradLoss()

        # Метрики
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                # Инициализация сверток
                if isinstance(m, nn.Conv2d):
                    if 'expert_heads' in name or 'orchestrator' in name:
                        # Делаем головы экспертов "тихими" на старте через Xavier
                        nn.init.xavier_uniform_(m.weight, gain=0.1)
                    else:
                        nn.init.kaiming_normal_(m.weight,
                                                mode="fan_out",
                                                nonlinearity="relu")

                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # Специальная инициализация для mu_offsets и w_init уже в __init__ модели
            print(
                f'HGSA_v14: Expert-Centric initialization complete. Unbound activations enabled.'
            )

    def configure_optimizers(self):
        # Группировка параметров для тонкой настройки LR
        params_groups = {
            'backbone': [],  # Encoder, Xi-Net
            'experts': [],  # ExpertHeads (v14 High-Freq Heads)
            'orchestra': [],  # Orchestrator
            'chi': [],  # Chi-Net
            'globals': [],  # mu_offsets, w_init, sigma_init
        }

        for name, param in self.model.named_parameters():
            if any(x in name for x in ['mu_offsets', 'w_init', 'sigma_init']):
                params_groups['globals'].append(param)
            elif 'expert_heads' in name:
                params_groups['experts'].append(param)
            elif 'orchestrator' in name:
                params_groups['orchestra'].append(param)
            elif 'chi_net' in name:
                params_groups['chi'].append(param)
            else:
                params_groups['backbone'].append(param)

        optimizer = optim.AdamW(
            [
                {
                    'params': params_groups['backbone'],
                    'lr': self.lr * 0.7
                },
                {
                    'params': params_groups['experts'],
                    'lr': self.lr * 1.2,
                    'weight_decay': 5e-3
                },  # Выше LR для разблокированных голов
                {
                    'params': params_groups['orchestra'],
                    'lr': self.lr * 0.8
                },
                {
                    'params': params_groups['chi'],
                    'lr': self.lr * 0.8
                },
                {
                    'params': params_groups['globals'],
                    'lr': self.lr * 0.5,
                    'weight_decay': 0.0
                },
            ],
            weight_decay=self.weight_decay)

        self.scheduler_switch_epoch = int(self.trainer.max_epochs * 0.8)
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs

        # OneCycle для разгона PSNR
        self.scheduler_1 = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.scheduler_switch_epoch,
            total_steps=self.scheduler_switch_epoch * steps_per_epoch,
            pct_start=0.15,
            div_factor=10,
            final_div_factor=100)

        # Доводка DeltaE
        self.scheduler_2 = optim.lr_scheduler.ExponentialLR(optimizer,
                                                            gamma=0.97)

        return {"optimizer": optimizer}

    def total_variation_loss(self, p_list):
        # Лосс на гладкость карт параметров Гауссиан
        loss = 0
        for p in p_list:
            # p: [B, C, 5, H, W] -> (w, mu, sigma, gate, tau)
            # Штрафуем резкие скачки mu и w, так как они теперь не ограничены tanh
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

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        main_out, usgs_out, p_list = self(src)

        # Клампим только для метрик
        main_out_c = torch.clamp(main_out, 0.0, 1.0)

        # 1. Основные лоссы (v14 делает упор на градиенты для четкости)
        loss_color = self.logcosh_loss(main_out, tgt)
        loss_grad = self.grad_loss(main_out, tgt)

        # 2. Структурный лосс
        ssim_val = self.ssim_metric(main_out_c, tgt)
        loss_ssim = 1.0 - torch.clamp(ssim_val, 0., 1.)

        # 3. Вспомогательные лоссы (v14 требует больше TV из-за unbound параметров)
        loss_aux = self.mae_loss(usgs_out, tgt)
        loss_tv = self.total_variation_loss(p_list)

        # 4. DeltaE фаза
        loss_de = 0.0
        if self.current_epoch >= self.scheduler_switch_epoch:
            loss_de = self.de_metric(main_out_c, tgt).mean() * 0.1

        # 5. Динамические веса
        if self.current_epoch < self.warmup_epochs:
            w_grad = 2.0
            w_aux = 1.0
        else:
            w_grad = 1.0
            w_aux = 0.1

        loss = (w_grad * loss_grad + 1.0 * loss_color + 0.5 * loss_ssim +
                w_aux * loss_aux + 0.05 * loss_tv + loss_de)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_psnr',
                 self.psnr_metric(main_out_c, tgt),
                 prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)
        y = torch.clamp(y, 0.0, 1.0)

        psnr_val = self.psnr_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)

        return de_val

    def forward(self, x):
        return self.model(src=x)['res']
