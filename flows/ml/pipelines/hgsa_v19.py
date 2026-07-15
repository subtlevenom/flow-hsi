import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import Dict, Tuple
from ..metrics import PSNR, SSIM, DeltaE

# ─────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────


class LogCoshLoss(nn.Module):

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        x = y_pred - y_true
        return torch.mean(
            torch.abs(x) + F.softplus(-2.0 * torch.abs(x)) - math.log(2.0))


class GradLoss(nn.Module):

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        def gradient(x):
            r = F.pad(x, (0, 1, 0, 1))
            gx = r[:, :, :, 1:] - r[:, :, :, :-1]
            gy = r[:, :, 1:, :] - r[:, :, :-1, :]
            return gx[:, :, :, :-1], gy[:, :, :-1, :]

        gx_p, gy_p = gradient(pred)
        gx_t, gy_t = gradient(target)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


class CIELabLoss(nn.Module):
    """Direct Delta-E optimization proxy in CIELAB space."""
    _XYZ_REF = torch.tensor([0.95047, 1.00000, 1.08883])
    _RGB2XYZ = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    def _rgb_to_xyz(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb_lin = torch.where(rgb <= 0.04045, rgb / 12.92,
                              ((rgb + 0.055) / 1.055)**2.4)
        m = self._RGB2XYZ.to(rgb.device)
        return torch.einsum('oi, bihw -> bohw', m, rgb_lin)

    def _xyz_to_lab(self, xyz: torch.Tensor) -> torch.Tensor:
        ref = self._XYZ_REF.to(xyz.device).view(1, 3, 1, 1)
        xyz_n = xyz / ref
        eps, kappa = 0.008856, 903.3
        f = torch.where(xyz_n > eps,
                        xyz_n.clamp(min=eps)**(1.0 / 3.0),
                        (kappa * xyz_n + 16.0) / 116.0)
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        L = (116.0 * fy - 16.0).unsqueeze(1)
        a = (500.0 * (fx - fy)).unsqueeze(1)
        b = (200.0 * (fy - fz)).unsqueeze(1)
        return torch.cat([L, a, b], dim=1)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        p_32 = pred.clamp(0, 1).to(torch.float32)
        t_32 = target.clamp(0, 1).to(torch.float32)

        pred_lab = self._xyz_to_lab(self._rgb_to_xyz(p_32))
        target_lab = self._xyz_to_lab(self._rgb_to_xyz(t_32))

        loss = F.l1_loss(pred_lab, target_lab)
        return loss.to(pred.dtype)


class FrequencyLoss(nn.Module):

    def __init__(self, loss_weight: float = 0.4):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pred_32 = pred.to(torch.float32)
        target_32 = target.to(torch.float32)

        pred_fft = torch.fft.rfft2(pred_32, norm='ortho')
        target_fft = torch.fft.rfft2(target_32, norm='ortho')

        loss = F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(
            pred_fft.imag, target_fft.imag)
        return self.loss_weight * loss.to(pred.dtype)


class MongeKantorovichLoss(nn.Module):

    def forward(self, src: torch.Tensor,
                transported: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(src, transported)


# ─────────────────────────────────────────────────────────────────────
# Pipeline v19-Pro
# ─────────────────────────────────────────────────────────────────────


class GEOTPipeline_v19_Pro(L.LightningModule):
    """
    SOTA Pipeline for HGSA_GEOT_v19_Pro.
    
    Strategy:
    1. Unleashed Transport: Reduced MK penalty (0.01) to allow wide color shifts.
    2. Perceptual Priority: High Lab weight (2.0) and Frequency weight (0.4).
    3. Structural Guard: Integrated SSIM loss (0.15) into the main objective.
    4. Fast Adaptation: 2.0x LR multiplier for GEOT transport heads.
    """

    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-3,
                 warmup_epochs: int = 15,
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

        # Weights (cmKAN-Killer Config)
        self.w_lab = 2.0  # Primary driver for < 4.51 dE
        self.w_freq = 1.0  # FrequencyLoss has its own internal 0.4 weight
        self.w_mk = 0.01  # Reduced: Allow the quadratic map to "stretch"
        self.w_ssim = 0.15  # Structural protection
        self.w_aux = 0.25  # Supervision on raw transport
        self.w_grad = 0.5

        self.save_hyperparameters(ignore=['model'])

    @property
    def model(self):
        # Adjust this path based on your wrapper structure
        return self._model.layers.hgsa

    def setup(self, stage: str = None) -> None:
        if stage != 'fit': return
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                if any(x in name for x in ['experts', 'net', 'film']):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear) and ('film' in name
                                               or 'conditioner' in name):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        groups = {'cond': [], 'enc': [], 'geot': [], 'fuse': []}
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if 'conditioner' in name: groups['cond'].append(param)
            elif 'encoder' in name: groups['enc'].append(param)
            elif 'geot_transport' in name: groups['geot'].append(param)
            else: groups['fuse'].append(param)

        optimizer = optim.AdamW(
            [
                {
                    'params': groups['cond'],
                    'lr': self.lr * 0.5
                },
                {
                    'params': groups['enc'],
                    'lr': self.lr * 0.8
                },
                {
                    'params': groups['geot'],
                    'lr': self.lr * 2.0,
                    'weight_decay': 1e-3
                },  # Faster manifold adaptation
                {
                    'params': groups['fuse'],
                    'lr': self.lr * 1.0
                },
            ],
            weight_decay=self.weight_decay)

        steps = self.trainer.estimated_stepping_batches
        self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=[
                                                           self.lr * 0.5,
                                                           self.lr * 0.8,
                                                           self.lr * 2.0,
                                                           self.lr * 1.0
                                                       ],
                                                       total_steps=steps,
                                                       pct_start=0.15,
                                                       div_factor=10,
                                                       final_div_factor=100)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'interval': 'step'
            }
        }

    def _compute_loss(self, src, main_out, aux_out, tgt, is_warmup):
        main_c = torch.clamp(main_out, 0, 1)
        aux_c = torch.clamp(aux_out, 0, 1)

        loss_mae = self.mae_loss(main_out, tgt)
        loss_aux = self.mae_loss(aux_out, tgt)

        if is_warmup:
            total = loss_mae + self.w_aux * loss_aux
            return total, {'loss_mae': loss_mae, 'loss_aux': loss_aux}

        # Main Phase: Perceptual + Structural + Transport
        loss_lab = self.lab_loss(main_c, tgt)
        loss_freq = self.freq_loss(main_c, tgt)
        loss_grad = self.grad_loss(main_c, tgt)
        loss_mk = self.mk_loss(src, aux_c)

        # Structural Guard: L1 + 0.15 * (1-SSIM)
        ssim_val = self.ssim_metric(main_c.to(torch.float32),
                                    tgt.to(torch.float32))
        loss_ssim = 1.0 - ssim_val.to(main_c.dtype)

        total = (loss_mae + self.w_ssim * loss_ssim + self.w_lab * loss_lab +
                 self.w_freq * loss_freq + self.w_grad * loss_grad +
                 self.w_aux * loss_aux + self.w_mk * loss_mk)

        return total, {
            'loss_mae': loss_mae,
            'loss_lab': loss_lab,
            'loss_freq': loss_freq,
            'loss_ssim': loss_ssim,
            'loss_mk': loss_mk
        }

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        main_out, aux_out = self(src)

        total_loss, details = self._compute_loss(
            src, main_out, aux_out, tgt, self.current_epoch
            < self.warmup_epochs)

        self.log('train_loss', total_loss, prog_bar=True)
        for k, v in details.items():
            self.log(f'train/{k}', v)
        return total_loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)

        psnr = self.psnr_metric(y, tgt)
        ssim = self.ssim_metric(y, tgt)
        de = self.de_metric(y, tgt).mean()

        # We monitor 'val_de' as the primary indicator for cmKAN competition
        self.log('val_de', de, prog_bar=True, sync_dist=True)
        self.log('val_psnr', psnr, prog_bar=True, sync_dist=True)
        self.log('val_ssim', ssim, sync_dist=True)
        self.log('val_loss', de, sync_dist=True)  # Checkpoint on dE

        return de

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)
        psnr = self.psnr_metric(y, tgt)
        ssim = self.ssim_metric(y, tgt)
        de = self.de_metric(y, tgt).mean()

        self.log('test_psnr', psnr)
        self.log('test_ssim', ssim)
        self.log('test_de', de)
        return de

    def forward(self, x):
        out = self.model(x)['res']
        if self.training: return out  # (final, aux)
        if isinstance(out, (tuple, list)):
            return torch.clamp(out[0], 0, 1)
        return torch.clamp(out, 0, 1)
