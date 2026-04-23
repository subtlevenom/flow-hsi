import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List


class HSGAPipeline_v8(L.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 optimizer: str = 'adamw',
                 lr: float = 2e-4,
                 warmup_epochs: int = 20,
                 weight_decay: float = 1e-4,
                 metrics_channels: List[int] = [0, 1, 2]) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.mae_loss = nn.L1Loss()
        self.aux_loss = nn.MSELoss()

        # Метрики (предполагается, что они импортированы из вашего модуля)
        from ..metrics import PSNR, SSIM, DeltaE
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        """ Инициализация v8: MSAB блоки и головы параметров """
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                # 1. Стандартная инициализация сверток
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # 2. Инициализация MSAB (Spectral Attention)
                # Важно: rescale в MS_MSA инициализируем единицами
                if 'rescale' in name:
                    nn.init.constant_(m, 1.0)

                # 3. Инициализация финальных проекций голов (v8 Encoder)
                # Делаем их очень маленькими, чтобы Гауссианы на старте были "нейтральными"
                if any(x in name for x in ['proj_mu', 'proj_sigma', 'proj_A']):
                    if hasattr(m, 'weight'):
                        nn.init.normal_(m.weight, mean=0.0, std=0.0001)

            print(
                f'HGSA_v8 Sandwich: Setup complete. MSAB-Heads initialized. Warmup: {self.warmup_epochs} epochs.'
            )

    def configure_optimizers(self):
        """ Распределение lr для v8: головы и энкодер теперь важнее """
        params_groups = {
            'xi': [],  # Xi-Net (аргумент)
            'encoder': [],  # Backbone + GCE
            'heads': [],  # MSAB-Heads (mu, sigma, A)
            'chi': [],  # Chi-Net (KAN)
            'aux': []  # usgs_to_img
        }

        for name, param in self.model.named_parameters():
            if 'xi_net' in name: params_groups['xi'].append(param)
            elif 'encoder.down' in name or 'encoder.gce' in name:
                params_groups['encoder'].append(param)
            elif 'head_' in name or 'proj_' in name:
                params_groups['heads'].append(param)
            elif 'chi_net' in name:
                params_groups['chi'].append(param)
            elif 'usgs_to_img' in name:
                params_groups['aux'].append(param)
            else:
                params_groups['encoder'].append(param)

        optimizer = optim.AdamW(
            [
                {
                    'params': params_groups['xi'],
                    'lr': self.lr * 0.5
                },
                {
                    'params': params_groups['encoder'],
                    'lr': self.lr
                },
                {
                    'params': params_groups['heads'],
                    'lr': self.lr * 1.2,
                    'weight_decay': 1e-2
                },  # Головам чуть больше lr
                {
                    'params': params_groups['aux'],
                    'lr': self.lr
                },
                {
                    'params': params_groups['chi'],
                    'lr': self.lr
                }
            ],
            weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def total_variation_loss(self, p_list):
        """ Регуляризация для v8 (p_list содержит тензоры [B, C, 5, H, W]) """
        loss = 0
        for p in p_list:
            # p[:, :, 1:3, :, :] -> mu, sigma
            # p[:, :, 3:, :, :] -> gate, tau
            params = p[:, :, 1:, :, :]
            diff_h = torch.abs(params[:, :, :, 1:, :] -
                               params[:, :, :, :-1, :])
            diff_w = torch.abs(params[:, :, :, :, 1:] -
                               params[:, :, :, :, :-1])
            loss += (diff_h.mean() + diff_w.mean())
        return loss / len(p_list) if p_list else 0

    def on_train_epoch_start(self):
        """ Warmup фаза для v8 """
        if self.current_epoch < self.warmup_epochs:
            for name, param in self.model.named_parameters():
                # На вармапе учим только "внешние" части: головы, KAN и вспомогательный выход
                if any(x in name for x in
                       ["encoder.head", "proj_", "usgs_to_img", "chi_net"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def on_before_optimizer_step(self, optimizer):
        # Градиентный клиппинг важен для MSAB
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def forward(self, x):
        # В v8 модель возвращает кортеж в режиме обучения и тензор в режиме валидации
        return self.model(src=x)['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        main_out, usgs_out, p_list = self.forward(src)

        # Ограничение диапазона
        main_out = torch.clamp(main_out, 0.0, 1.0)
        usgs_out = torch.clamp(usgs_out, 0.0, 1.0)

        # Лоссы
        loss_mae = self.mae_loss(main_out, tgt)
        ssim_val = self.ssim_metric(main_out, tgt)
        loss_ssim = 1.0 - torch.clamp(ssim_val, 0., 1.)
        loss_aux = self.aux_loss(usgs_out, tgt)
        loss_tv = self.total_variation_loss(p_list)

        # Баланс лоссов для v8 (чуть больше веса на aux для dE)
        loss = 1.0 * loss_mae + 0.5 * loss_ssim + 0.4 * loss_aux + 0.05 * loss_tv

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
        y = self.forward(src)  # В режиме eval вернет только main_out

        y = torch.clamp(y, 0.0, 1.0)

        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)
        self.log('val_loss', de_val, prog_bar=True)

        return psnr_val

    def predict_step(self, batch, batch_idx):
        src, tgt = batch
        return self(src)
