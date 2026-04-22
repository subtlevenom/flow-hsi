import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import List

from ..metrics import (PSNR, SSIM, DeltaE)


class HSGAPipeline(L.LightningModule):

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

        self.psnr_metric = PSNR(data_range=(0, 1))
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        """ 
        Инициализация весов. 
        Важно: USGS головы инициализируются около нуля, чтобы начать с 'чистого листа'.
        """
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                # Стандартная инициализация сверток и линейных слоев
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # Инициализация голов параметров USGS (v7 Enhanced)
                # Малое стандартное отклонение предотвращает взрыв гауссиан на старте
                if 'param_heads' in name and hasattr(m, 'weight'):
                    nn.init.normal_(m.weight, mean=0.0, std=0.0001)

            print(
                f'HGSA_v7 Enhanced: Setup complete. Warmup: {self.warmup_epochs} epochs.'
            )

    def configure_optimizers(self):
        xi_params, encoder_params, param_heads_params, chi_params, usgs_aux_params = [], [], [], [], []
        for name, param in self.model.named_parameters():
            if 'xi_net' in name: xi_params.append(param)
            elif 'encoder' in name: encoder_params.append(param)
            elif 'param_heads' in name: param_heads_params.append(param)
            elif 'chi_net' in name: chi_params.append(param)
            elif 'usgs_to_img' in name: usgs_aux_params.append(param)
            else: encoder_params.append(param)

        optimizer = optim.AdamW([{
            'params': xi_params,
            'lr': self.lr * 0.5
        }, {
            'params': encoder_params,
            'lr': self.lr
        }, {
            'params': param_heads_params,
            'lr': self.lr,
            'weight_decay': 1e-2
        }, {
            'params': usgs_aux_params,
            'lr': self.lr
        }, {
            'params': chi_params,
            'lr': self.lr
        }],
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

    def total_variation_loss(self, p_map_list):
        """ Штрафуем за резкие изменения mu, sigma, gate, tau в пространстве """
        loss = 0
        for p_map in p_map_list:
            # p_map: [B, C, 5, H, W]
            # Берем срезы параметров со 2-го по 5-й (mu, sigma, gate, tau)
            # w (индекс 0) обычно не штрафуют TV, чтобы оставить резкость деталей
            params = p_map[:, :, 1:, :, :]

            diff_h = torch.abs(params[:, :, :, 1:, :] -
                               params[:, :, :, :-1, :])
            diff_w = torch.abs(params[:, :, :, :, 1:] -
                               params[:, :, :, :, :-1])
            loss += (diff_h.mean() + diff_w.mean())
        return loss / len(p_map_list)

    def on_train_epoch_start(self):
        if self.current_epoch < self.warmup_epochs:
            for name, param in self.model.named_parameters():
                if any(x in name
                       for x in ["param_heads", "usgs_to_img", "chi_net"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def forward(self, x):
        return self.model(src=x)['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # Получаем 3 выхода от модели
        main_out, usgs_out, p_list = self(src)

        # 1. КРИТИЧЕСКИ ВАЖНО: Ограничиваем выход перед расчетом лоссов
        # Это гарантирует, что SSIM и MAE будут адекватными
        main_out = torch.clamp(main_out, 0.0, 1.0)
        usgs_out = torch.clamp(usgs_out, 0.0, 1.0)

        # 1. Основной лосс
        loss_mae = self.mae_loss(main_out, tgt)
        ssim_loss = self.ssim_metric(main_out, tgt)
        loss_ssim = 1.0 - torch.clamp(ssim_loss, 0., 1.)

        # 2. Auxiliary лосс (прямое обучение Гауссиан)
        loss_aux = self.aux_loss(usgs_out, tgt)

        # 3. TV Loss (регуляризация параметров)
        loss_tv = self.total_variation_loss(p_list)

        # Итоговый баланс
        # Коэффициент 0.05 для TV достаточно мал, чтобы не "замылить" всё,
        # но достаточно велик, чтобы убрать шум в картах параметров.
        loss = 1.0 * loss_mae + 0.5 * loss_ssim + 0.3 * loss_aux + 0.05 * loss_tv

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_aux', loss_aux, prog_bar=False)
        self.log('train_tv', loss_tv, prog_bar=False)

        with torch.no_grad():
            psnr_val = self.psnr_metric(main_out, tgt)
            self.log('train_mae', loss_mae, prog_bar=True)
            self.log('train_psnr', psnr_val, prog_bar=True)
            self.log('train_ssim', ssim_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)
        mae_val = self.mae_loss(y, tgt)
        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()
        self.log('val_mae', mae_val, prog_bar=True)
        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)
        self.log('val_loss', de_val, prog_bar=True)
        return mae_val

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)
        psnr_val = self.psnr_metric(y, tgt)
        self.log('test_psnr', psnr_val)
        return psnr_val

    def predict_step(self, batch, batch_idx):
        src, tgt = batch
        return self(src)
