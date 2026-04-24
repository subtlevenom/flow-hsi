import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List

from ..metrics import PSNR, SSIM, DeltaE


class HSGAPipeline_v11(L.LightningModule):

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

        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Метрики
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        """ Инициализация v11 Smart Orchestra """
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                # 1. Стандартная инициализация для MSAB и сверток
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # 2. Инициализация проекционных слоев в головах (делаем их "тихими")
                if any(x in name for x in
                       ['expert_heads', 'orchestrator_proj', 'chi_post']):
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, mean=0.0, std=0.0001)

            print(
                f'HGSA_v11 Smart Orchestra: MSAB blocks and Expert FFNs initialized. Warmup: {self.warmup_epochs} epochs.'
            )

    def configure_optimizers(self):
        """ Дифференцированный LR для v11: MSAB блоки требуют более тонкой настройки """
        params_groups = {
            'backbone': [],  # Encoder2D + Xi-Net
            'experts': [],  # Expert Heads (FFN)
            'orchestra': [],  # Orchestrator (MSAB + Proj)
            'chi': [],  # Chi-Net (MSAB + Proj)
            'aux': []  # usgs_to_img
        }

        for name, param in self.model.named_parameters():
            if 'xi_net' in name or 'encoder' in name:
                params_groups['backbone'].append(param)
            elif 'expert_heads' in name:
                params_groups['experts'].append(param)
            elif 'orchestrator' in name:
                params_groups['orchestra'].append(param)
            elif 'chi_' in name:  # chi_pre, chi_msab, chi_post
                params_groups['chi'].append(param)
            elif 'usgs_to_img' in name:
                params_groups['aux'].append(param)
            else:
                params_groups['backbone'].append(param)

        optimizer = optim.AdamW(
            [
                {
                    'params': params_groups['backbone'],
                    'lr': self.lr * 0.5
                },
                {
                    'params': params_groups['experts'],
                    'lr': self.lr * 1.0,
                    'weight_decay': 1e-2
                },
                {
                    'params': params_groups['orchestra'],
                    'lr': self.lr * 0.8
                },  # MSAB учим чуть медленнее FFN
                {
                    'params': params_groups['chi'],
                    'lr': self.lr * 0.8
                },
                {
                    'params': params_groups['aux'],
                    'lr': self.lr
                }
            ],
            weight_decay=self.weight_decay)

        # Schedulers

        self.scheduler_switch_epoch = int(self.trainer.max_epochs * 5./3.)

        FINAL_LR_ONECYCLE = self.lr / 200
        DIV_FACTOR=10
        FINAL_DIV_FACTOR = self.lr / (FINAL_LR_ONECYCLE * DIV_FACTOR)
        GAMMA = 0.9885

        # 2. Первый планировщик: OneCycleLR (0 -> 320 эпохи)
        # pct_start: доля времени на подъем LR (обычно 0.3)
        # div_factor: начальный LR = max_lr / div_factor
        # final_div_factor: финальный LR = initial_lr / final_div_factor
        self.scheduler_1 = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.scheduler_switch_epoch,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.15,    # Увеличен разогрев для стабильности Гауссиан
            div_factor=DIV_FACTOR,     # Начальный LR не слишком низкий (2e-5)
            final_div_factor=FINAL_DIV_FACTOR # Финальный LR не дает модели "заснуть" (4e-6)
        )
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
            # optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7)

        # 3. Второй планировщик: ExponentialLR (320 -> 520 эпохи)
        self.scheduler_2 = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=GAMMA,
        )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
                # "scheduler": scheduler,
                # "interval": "epoch"
            # }
        }

    def total_variation_loss(self, p_list):
        """ Регуляризация параметров Гауссиан (w, mu, sigma, gate, tau) """
        loss = 0
        for p in p_list:
            # p: [B, C, 5, H, W]
            # Применяем TV ко всем параметрам для плавности цветовых карт
            diff_h = torch.abs(p[:, :, :, 1:, :] - p[:, :, :, :-1, :])
            diff_w = torch.abs(p[:, :, :, :, 1:] - p[:, :, :, :, :-1])
            loss += (diff_h.mean() + diff_w.mean())
        return loss / len(p_list) if p_list else 0

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if self.current_epoch < self.scheduler_switch_epoch:
            self.scheduler_1.step()
        else:
            # После 320 эпохи делаем шаг экспоненты раз в эпоху
            if self.trainer.is_last_batch:
                self.scheduler_2.step()

    def on_train_epoch_start(self):
        """ Фазовый прогрев: Сначала эксперты и MSAB-интеграторы, потом Backbone """
        if self.current_epoch < self.warmup_epochs:
            for name, param in self.model.named_parameters():
                if any(
                        x in name for x in
                    ["expert_heads", "orchestrator", "chi_", "usgs_to_img"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def on_before_optimizer_step(self, optimizer):
        # MSAB может давать всплески градиента, клиппинг обязателен
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

    def forward(self, x):
        return self.model(src=x)['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # v11 возвращает (main_out, usgs_out, p_list)
        main_out, usgs_out, p_list = self.forward(src)

        main_out = torch.clamp(main_out, 0.0, 1.0)
        usgs_out = torch.clamp(usgs_out, 0.0, 1.0)

        loss_mae = self.mae_loss(main_out, tgt)

        ssim_val = self.ssim_metric(main_out, tgt)
        loss_ssim = 1.0 - torch.clamp(ssim_val, 0., 1.)

        loss_aux = self.mae_loss(usgs_out, tgt)
        loss_tv = self.total_variation_loss(p_list)

        # Обновленная формула лосса v11 (Сбалансированная)
        # 1.0*MAE (Цвет) + 0.2*SSIM (Структура) + 0.1*AUX (Гауссианы) + 0.01*TV (Плавность)
        loss = 1.0 * loss_mae + 0.2 * loss_ssim + 0.1 * loss_aux + 0.01 * loss_tv

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_aux', loss_aux, prog_bar=False)

        with torch.no_grad():
            psnr_val = self.psnr_metric(main_out, tgt)
            self.log('train_mae', loss_mae, prog_bar=True)
            self.log('train_psnr', psnr_val, prog_bar=True)
            self.log('train_ssim', ssim_val, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self.forward(src)
        y = torch.clamp(y, 0.0, 1.0)

        mae_val = self.mae_loss(y, tgt)
        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('val_mae', mae_val, prog_bar=True)
        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)
        self.log('val_loss', de_val, prog_bar=True)

        return de_val

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self.forward(src)
        y = torch.clamp(y, 0.0, 1.0)

        mae_val = self.mae_loss(y, tgt)
        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('test_mae', mae_val, prog_bar=True)
        self.log('test_psnr', psnr_val, prog_bar=True)
        self.log('test_ssim', ssim_val, prog_bar=True)
        self.log('test_de', de_val, prog_bar=True)

        return de_val

    def predict_step(self, batch, batch_idx):
        src, tgt = batch
        return torch.clamp(self.model(src), 0.0, 1.0)
