import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List
from flows.tools.utils import models

from ..metrics import PSNR, SSIM, DeltaE


# --- Loss Functions ---
class LogCoshLoss(nn.Module):

    def forward(self, y_pred, y_true):
        x = y_pred - y_true
        loss = torch.abs(x) + F.softplus(-2. * torch.abs(x)) - math.log(2.0)
        return torch.mean(loss)


class GradLoss(nn.Module):

    def forward(self, pred, target):
        grads_pred = torch.gradient(pred, dim=(2, 3))
        grads_tgt = torch.gradient(target, dim=(2, 3))
        return F.l1_loss(grads_pred[0], grads_tgt[0]) + F.l1_loss(
            grads_pred[1], grads_tgt[1])


# --- Pipeline HGSA v15 ---
class HSGAPipeline_v15(L.LightningModule):

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

        self.mae_loss = nn.L1Loss()
        self.logcosh_loss = LogCoshLoss()
        self.grad_loss = GradLoss()

        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        # Коэффициенты регуляризации
        self.w_repulsion = 0.01  # Сила отталкивания центров
        self.w_sigma_reg = 0.01  # Контроль ширины Гауссиан

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d):
                    # Специальная инициализация для голов и словаря базисов
                    if any(x in name for x in
                           ['expert_heads', 'orchestrator', 'q_proj']):
                        nn.init.xavier_uniform_(m.weight, gain=0.1)
                    else:
                        nn.init.kaiming_normal_(m.weight,
                                                mode="fan_out",
                                                nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # Инициализация обучаемого словаря BasisAttention
                if 'basis_v' in name:
                    # Инициализируем равномерно, чтобы на старте базис был адекватным
                    nn.init.uniform_(m, 0.0, 1.0)

            MODEL_PATH = '.experiments/ggpd.hgsa_v15.huawei/logs/checkpoints/_last.ckpt'
            models.load_model(self.model, 'model', MODEL_PATH)

            print(
                f'HGSA_v15: Hybrid Parametric-Attention USGS initialization complete.'
            )

    def configure_optimizers(self):

        params_groups = {
            'backbone': [],
            'basis': [],
            'experts': [],
            'orchestra': [],
            'fusion': [],  # Новое для v15
            'globals': [],
        }

        for name, param in self.model.named_parameters():
            # Исправленные имена параметров для v15
            if any(x in name
                   for x in ['mu_min', 'mu_max', 'w_init', 'sigma_init']):
                params_groups['globals'].append(param)
            elif 'xi_net' in name or 'basis_v' in name:
                params_groups['basis'].append(param)
            elif 'expert_heads' in name:
                params_groups['experts'].append(param)
            elif 'orchestrator' in name:
                params_groups['orchestra'].append(param)
            elif 'fusion' in name or 'chi_net' in name:  # FusionNet и LightChiNet
                params_groups['fusion'].append(param)
            else:
                params_groups['backbone'].append(param)

        optimizer = optim.AdamW(
            [
                {
                    'params': params_groups['backbone'],
                    'lr': self.lr * 0.7
                },
                {
                    'params': params_groups['basis'],
                    'lr': self.lr * 0.5
                },
                {
                    'params': params_groups['experts'],
                    'lr': self.lr * 1.2,
                    'weight_decay': 8e-3
                },
                {
                    'params': params_groups['orchestra'],
                    'lr': self.lr * 0.8
                },
                {
                    'params': params_groups['fusion'],
                    'lr': self.lr * 1.0
                },  # Fusion учим активно
                {
                    'params': params_groups['globals'],
                    'lr': self.lr * 0.5,
                    'weight_decay': 0.0
                },
            ],
            weight_decay=self.weight_decay)

        self.scheduler_switch_epoch = int(self.trainer.max_epochs * 0.8)
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs

        # 2. Replace OneCycleLR with CosineAnnealingWarmRestarts
        # We want 3 cycles within the scheduler_1 window.
        # Total steps for scheduler_1 = self.scheduler_switch_epoch * steps_per_epoch
        # T_0 = (Total steps) / 3
        total_steps_s1 = self.scheduler_switch_epoch * steps_per_epoch
        t_0_steps = total_steps_s1 // 3

        self.scheduler_1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0_steps,
            T_mult=1,
            eta_min=self.lr * 0.01 # Floor at 1% of max LR
        )

        self.scheduler_2 = optim.lr_scheduler.ExponentialLR(optimizer,
                                                            gamma=0.97)

        return {"optimizer": optimizer}

    def total_variation_loss(self, psi_total):
        # В v15 мы штрафуем резкость итогового поля проекций (psi_total)
        # psi_total: [B, Q*3, H, W]
        diff_h = torch.abs(psi_total[:, :, 1:, :] - psi_total[:, :, :-1, :])
        diff_w = torch.abs(psi_total[:, :, :, 1:] - psi_total[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()

    def gaussian_stability_loss(self):
        """ Обновлено для v15: следим за mu_min/max и sigma """
        repulsion_loss = 0.0
        sigma_loss = 0.0

        for m in self.model.modules():
            if hasattr(m, 'mu_min') and hasattr(m, 'mu_max'):
                # Штрафуем за слишком близкие mu_min и mu_max в одном канале
                # Мы хотим, чтобы mu_max - mu_min было ощутимым
                dist = torch.abs(m.mu_max - m.mu_min)
                repulsion_loss += torch.exp(-5.0 * dist).mean()

            if hasattr(m, 'sigma_init'):
                sigma_loss += torch.mean(torch.abs(m.sigma_init))

        return repulsion_loss, sigma_loss

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        if self.current_epoch < self.scheduler_switch_epoch:
            self.scheduler_1.step()
        else:
            if self.trainer.is_last_batch:
                self.scheduler_2.step()

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        # В v15 модель возвращает (usgs_out, sagf_out) в режиме training
        # sagf_out — это x + aux_proj(psi_total)
        main_out, sagf_out = self(src)

        # 1. Основные лоссы
        loss_color = self.logcosh_loss(main_out, tgt)
        loss_grad = self.grad_loss(main_out, tgt)

        # 2. Структурный лосс
        main_out_c = torch.clamp(main_out, 0.0, 1.0)
        ssim_val = self.ssim_metric(main_out_c, tgt)
        loss_ssim = 1.0 - torch.clamp(ssim_val, 0., 1.)

        # 3. Вспомогательные лоссы (v15: используем sagf_out вместо usgs_out)
        loss_aux = self.mae_loss(sagf_out, tgt)
        # 4.Регуляризация стабильности экспертов
        loss_repulsion, loss_sigma_reg = self.gaussian_stability_loss()

        # 5. TV Loss на поле проекций (нужно прокинуть psi_total или считать на sagf_out)
        # Для v15 упростим: считаем TV на самом предсказании для подавления артефактов
        loss_tv = self.total_variation_loss(main_out)

        # 6. DeltaE фаза
        loss_de = 0.0
        # if self.current_epoch >= self.scheduler_switch_epoch:
        # loss_de = self.de_metric(main_out_c, tgt).mean() * 0.1

        # Динамические веса
        w_grad = 2.0 if self.current_epoch < self.warmup_epochs else 1.0
        w_aux = 1.0 if self.current_epoch < self.warmup_epochs else 0.1

        loss = (
            w_grad * loss_grad + 1.0 * loss_color + 0.5 * loss_ssim +
            w_aux * loss_aux + 0.05 * loss_tv + loss_de +
            self.w_repulsion * loss_repulsion +  # Отталкивание
            self.w_sigma_reg * loss_sigma_reg)  # Компактность

        self.log('train_loss', loss, prog_bar=True)
        self.log('loss_rep', loss_repulsion, prog_bar=False)

        with torch.no_grad():
            psnr_val = self.psnr_metric(main_out_c, tgt)
            self.log('train_psnr', psnr_val, prog_bar=True)
            self.log('train_ssim', ssim_val, prog_bar=True)
            self.log('train_de', loss_de, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)  # В режиме eval возвращает только один тензор
        y = torch.clamp(y, 0.0, 1.0)

        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)
        self.log('val_loss', de_val, prog_bar=True)
        return de_val

    def forward(self, x):
        return self.model(src=x)['res']
