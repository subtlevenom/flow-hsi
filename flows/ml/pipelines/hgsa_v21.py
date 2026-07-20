import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import Dict, Tuple
from ..metrics import PSNR, SSIM, DeltaE

# ─────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

class LogCoshLoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        x = y_pred - y_true
        return torch.mean(torch.abs(x) + F.softplus(-2.0 * torch.abs(x)) - math.log(2.0))

class GradLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def gradient(x):
            r = F.pad(x, (0, 1, 0, 1))
            gx = r[:, :, :, 1:] - r[:, :, :, :-1]
            gy = r[:, :, 1:, :] - r[:, :, :-1, :]
            return gx[:, :, :, :-1], gy[:, :, :-1, :]
        gx_p, gy_p = gradient(pred)
        gx_t, gy_t = gradient(target)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

class CIELabLoss(nn.Module):
    """Прямая оптимизация Delta-E в пространстве CIELAB."""
    _XYZ_REF = torch.tensor([0.95047, 1.00000, 1.08883])
    _RGB2XYZ = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    def _rgb_to_xyz(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb_lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055)**2.4)
        m = self._RGB2XYZ.to(rgb.device)
        return torch.einsum('oi, bihw -> bohw', m, rgb_lin)

    def _xyz_to_lab(self, xyz: torch.Tensor) -> torch.Tensor:
        ref = self._XYZ_REF.to(xyz.device).view(1, 3, 1, 1)
        xyz_n = xyz / ref
        eps, kappa = 0.008856, 903.3
        f = torch.where(xyz_n > eps, xyz_n.clamp(min=eps)**(1.0 / 3.0), (kappa * xyz_n + 16.0) / 116.0)
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        L = (116.0 * fy - 16.0).unsqueeze(1)
        a = (500.0 * (fx - fy)).unsqueeze(1)
        b = (200.0 * (fy - fz)).unsqueeze(1)
        return torch.cat([L, a, b], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_32 = pred.clamp(0, 1).to(torch.float32)
        t_32 = target.clamp(0, 1).to(torch.float32)
        pred_lab = self._xyz_to_lab(self._rgb_to_xyz(p_32))
        target_lab = self._xyz_to_lab(self._rgb_to_xyz(t_32))
        return F.l1_loss(pred_lab, target_lab).to(pred.dtype)

class FrequencyLoss(nn.Module):
    def __init__(self, loss_weight: float = 0.4):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred.to(torch.float32), norm='ortho')
        target_fft = torch.fft.rfft2(target.to(torch.float32), norm='ortho')
        loss = F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(pred_fft.imag, target_fft.imag)
        return self.loss_weight * loss.to(pred.dtype)

class MongeKantorovichLoss(nn.Module):
    """Регуляризация 'расстояния' транспорта от источника."""
    def forward(self, src: torch.Tensor, transported: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(src, transported)

# ─────────────────────────────────────────────────────────────────────
# PIPELINE v21 (Updated for PixelWise USGS+EOT)
# ─────────────────────────────────────────────────────────────────────

class GEOTPipeline_v21(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-3,
                 warmup_epochs: int = 10,
                 weight_decay: float = 1e-4) -> None:
        super().__init__()
        self._model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Metrics
        self.psnr_metric = PSNR(data_range=1.0)
        self.ssim_metric = SSIM(data_range=1.0)
        self.de_metric = DeltaE()
        
        # Losses
        self.mae_loss = nn.L1Loss()
        self.lab_loss = CIELabLoss()
        self.freq_loss = FrequencyLoss(loss_weight=0.4)
        self.grad_loss = GradLoss()
        self.mk_loss = MongeKantorovichLoss()

        # Weights (v21 Optimized)
        self.w_lab = 2.5   # Усилен акцент на цветопередачу
        self.w_freq = 1.0
        self.w_mk = 0.05   # Регуляризация моста Шрёдингера
        self.w_ssim = 0.2
        self.w_aux = 0.5   # Вес для барицентрической проекции (aux_out)
        self.w_grad = 0.5

        self.save_hyperparameters(ignore=['model'])

    @property
    def model(self):
        # Доступ к HGSA_USGS_EOT_v21 через обертку
        return self._model.layers.hgsa

    def setup(self, stage: str = None) -> None:
        if stage != 'fit': return
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                # Инициализация гиперсетей USGS и транспортных голов
                if any(x in name for x in ['hyper', 'transport_head', 'phi_net']):
                    nn.init.xavier_uniform_(m.weight, gain=0.02)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        # Группировка параметров для v21
        groups = {'cond': [], 'enc': [], 'eot': [], 'fuse': []}
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if 'conditioner' in name: groups['cond'].append(param)
            elif 'encoder' in name: groups['enc'].append(param)
            elif 'eot_usgs' in name: groups['eot'].append(param) # Ядро USGS + T_q
            else: groups['fuse'].append(param)

        optimizer = optim.AdamW([
            {'params': groups['cond'], 'lr': self.lr * 0.5},
            {'params': groups['enc'], 'lr': self.lr * 0.8},
            {'params': groups['eot'], 'lr': self.lr * 1.5, 'weight_decay': 1e-3}, # Высокий LR для адаптации ядер
            {'params': groups['fuse'], 'lr': self.lr * 1.0},
        ], weight_decay=self.weight_decay)

        steps = self.trainer.estimated_stepping_batches
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.lr * 0.5, self.lr * 0.8, self.lr * 1.5, self.lr * 1.0],
            total_steps=steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def forward(self, x):
        res = self.model(x)['res']
        if self.training: return res # (final_out, transported_x)
        return torch.clamp(res[0] if isinstance(res, tuple) else res, 0, 1)

    def _compute_loss(self, src, main_out, aux_out, tgt, is_warmup):
        # main_out: результат после Fusion, aux_out: барицентрическая проекция T(x)
        loss_mae = self.mae_loss(main_out, tgt)
        loss_aux = self.mae_loss(aux_out, tgt)

        if is_warmup:
            return loss_mae + self.w_aux * loss_aux, {'mae': loss_mae, 'aux': loss_aux}

        m_c, a_c = torch.clamp(main_out, 0, 1), torch.clamp(aux_out, 0, 1)
        
        loss_lab = self.lab_loss(m_c, tgt)
        loss_freq = self.freq_loss(m_c, tgt)
        loss_grad = self.grad_loss(m_c, tgt)
        loss_mk = self.mk_loss(src, a_c) # Регуляризация транспорта

        total = (loss_mae + self.w_lab * loss_lab + self.w_freq * loss_freq + 
                 self.w_grad * loss_grad + self.w_aux * loss_aux + self.w_mk * loss_mk)

        return total, {'mae': loss_mae, 'lab': loss_lab, 'mk': loss_mk}

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        main_out, aux_out = self(src)
        loss, details = self._compute_loss(src, main_out, aux_out, tgt, self.current_epoch < self.warmup_epochs)
        self.log('train_loss', loss, prog_bar=True)
        for k, v in details.items(): self.log(f'train/{k}', v)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)

        loss_mae = F.l1_loss(y, tgt)
        psnr = self.psnr_metric(y, tgt)
        ssim = self.ssim_metric(y, tgt)
        de = self.de_metric(y, tgt).mean()

        self.log('val_mae', loss_mae, prog_bar=True, sync_dist=True)
        self.log('val_psnr', psnr)
        self.log('val_ssim', ssim)
        self.log('val_de', de)

        return loss_mae

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)

        loss_mae = F.l1_loss(y, tgt)
        psnr = self.psnr_metric(y, tgt)
        ssim = self.ssim_metric(y, tgt)
        de = self.de_metric(y, tgt).mean()

        self.log('test_mae', loss_mae, prog_bar=True, sync_dist=True)
        self.log('test_psnr', psnr)
        self.log('test_ssim', ssim)
        self.log('test_de', de)

        return loss_mae