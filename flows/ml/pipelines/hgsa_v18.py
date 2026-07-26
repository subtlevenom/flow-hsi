import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import Dict, Tuple
from flows.core import Logger
from flows.tools.utils import models
from flows.tools.utils.colors import rgb_to_lab
from ..metrics import PSNR, SSIM, DeltaE

# ─────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────


class LogCoshLoss(nn.Module):
    """Smooth L1-like loss, robust to outliers. Kept from v17."""

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        x = y_pred - y_true
        return torch.mean(
            torch.abs(x) + F.softplus(-2.0 * torch.abs(x)) - math.log(2.0))


class GradLoss(nn.Module):
    """Spatial gradient penalty for edge preservation. Kept from v17."""

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
    """
    NEW: Direct Delta-E optimization in CIELAB space.

    This is the single most impactful loss change for beating
    cmKAN's 4.51 dE. Standard L1/LogCosh in RGB space does NOT
    minimize perceptual color error — CIELAB does.

    Pipeline:
        RGB [0,1] → XYZ (linear) → CIELAB → L1 in Lab space

    The L1 in Lab space is a first-order approximation of Delta-E 76.
    We use L1 rather than L2 because Lab outliers (saturated colors)
    should not dominate the gradient.
    """
    # D65 illuminant reference white
    _XYZ_REF = torch.tensor([0.95047, 1.00000, 1.08883])

    # sRGB → XYZ (D65) matrix
    _RGB2XYZ = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    def __init__(self):
        super().__init__()
        self.register_buffer = lambda n, t: None  # buffers set in forward

    def _rgb_to_xyz(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: [B, 3, H, W] in [0, 1]
        # Linearize sRGB (approximate gamma)
        rgb_lin = torch.where(rgb <= 0.04045, rgb / 12.92,
                              ((rgb + 0.055) / 1.055)**2.4)
        m = self._RGB2XYZ.to(rgb.device)  # [3, 3]
        # Einsum: matrix multiply over channel dim
        xyz = torch.einsum('oi, bihw -> bohw', m, rgb_lin)
        return xyz

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
        return torch.cat([L, a, b], dim=1)  # [B, 3, H, W]

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pred_lab = self._xyz_to_lab(self._rgb_to_xyz(pred.clamp(0, 1)))
        target_lab = self._xyz_to_lab(self._rgb_to_xyz(target.clamp(0, 1)))
        # Weight chroma (a, b) above luminance (L). dE2000 — the reported
        # metric — penalizes chroma error more than a uniform (dE76-style)
        # Lab-L1 does, so this aligns the loss with the evaluation metric.
        w = torch.tensor([1.0, 2.0, 2.0], device=pred_lab.device,
                         dtype=pred_lab.dtype).view(1, 3, 1, 1)
        return ((pred_lab - target_lab).abs() * w).mean()


class FrequencyLoss(nn.Module):
    """
    NEW: FFT-based frequency reconstruction loss.

    Borrowed from FocalFrequencyLoss (ICCV 2021).
    Penalizes errors in the Fourier spectrum, which forces the model
    to reconstruct high-frequency texture (fine detail, edges) that
    spatial L1 loss tends to ignore.

    This directly addresses the texture-blurring tendency of
    Gaussian-kernel-based USGS manifolds.

    We use L1 on the complex spectrum (real + imag separately)
    rather than magnitude only, to preserve phase (= spatial structure).
    """

    def __init__(self, loss_weight: float = 0.1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # FFT ops don't support half/bfloat16 → compute in float32
        pred_fft = torch.fft.rfft2(pred.float(), norm='ortho')
        target_fft = torch.fft.rfft2(target.float(), norm='ortho')
        # L1 on real and imaginary parts separately
        loss = (F.l1_loss(pred_fft.real, target_fft.real) +
                F.l1_loss(pred_fft.imag, target_fft.imag))
        return (self.loss_weight * loss).to(pred.dtype)

class DeltaE2000Loss(nn.Module):
    """
    Differentiable CIEDE2000 loss.

    Optimizes the exact metric that is reported/checkpointed (DeltaE),
    unlike CIELabLoss which is only a dE76-style Lab-L1 proxy. The
    formula mirrors flows/ml/metrics/delta_e_2000.py, but every
    sqrt()/atan2() that can hit a 0/0 singularity — identical or
    neutral-gray pixels, both common on Volga2K — is epsilon-guarded so
    gradients stay finite. Computed in float32 for stability under bf16.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        lab1 = rgb_to_lab(pred.clamp(0, 1).float())
        lab2 = rgb_to_lab(target.clamp(0, 1).float())
        L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
        L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

        # chroma (guarded norms — hypot has NaN grad at the origin)
        C1 = torch.sqrt(a1 * a1 + b1 * b1 + eps)
        C2 = torch.sqrt(a2 * a2 + b2 * b2 + eps)
        C_bar = (C1 + C2) / 2
        C_bar7 = C_bar ** 7
        G = 0.5 * (1 - torch.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7) + eps))

        a1p = (1 + G) * a1
        a2p = (1 + G) * a2
        C1p = torch.sqrt(a1p * a1p + b1 * b1 + eps)
        C2p = torch.sqrt(a2p * a2p + b2 * b2 + eps)

        # hue angles in degrees (nudge x by eps so atan2 is never 0/0)
        h1p = torch.rad2deg(torch.atan2(b1, a1p + eps)) % 360
        h2p = torch.rad2deg(torch.atan2(b2, a2p + eps)) % 360

        dLp = L2 - L1
        dCp = C2p - C1p

        dhp = h2p - h1p
        dhp = torch.where(dhp > 180, dhp - 360, dhp)
        dhp = torch.where(dhp < -180, dhp + 360, dhp)
        dHp = 2 * torch.sqrt(C1p * C2p + eps) * torch.sin(
            torch.deg2rad(dhp / 2))

        Lbarp = (L1 + L2) / 2
        Cbarp = (C1p + C2p) / 2

        hsum = h1p + h2p
        habs = torch.abs(h1p - h2p)
        hbar = torch.where(
            habs <= 180, hsum / 2,
            torch.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2))

        T = (1
             - 0.17 * torch.cos(torch.deg2rad(hbar - 30))
             + 0.24 * torch.cos(torch.deg2rad(2 * hbar))
             + 0.32 * torch.cos(torch.deg2rad(3 * hbar + 6))
             - 0.20 * torch.cos(torch.deg2rad(4 * hbar - 63)))

        dtheta = 30 * torch.exp(-(((hbar - 275) / 25) ** 2))
        Cbarp7 = Cbarp ** 7
        RC = 2 * torch.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7) + eps)
        Lm = (Lbarp - 50) ** 2
        SL = 1 + (0.015 * Lm) / torch.sqrt(20 + Lm)
        SC = 1 + 0.045 * Cbarp
        SH = 1 + 0.015 * Cbarp * T
        RT = -torch.sin(torch.deg2rad(2 * dtheta)) * RC

        dE2 = ((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
               + RT * (dCp / SC) * (dHp / SH))
        return torch.sqrt(dE2.clamp(min=0.0) + eps).mean()

# ─────────────────────────────────────────────────────────────────────
# Pipeline v18
# ─────────────────────────────────────────────────────────────────────


class HSGAPipeline_v18(L.LightningModule):
    """
    Training pipeline for HGSA_v18.

    Key changes vs v17 pipeline:
    1. CIELabLoss  — direct Delta-E optimization (most impactful for dE)
    2. FrequencyLoss — FFT texture reconstruction (compensates for
                       Gaussian manifold smoothing)
    3. Loss schedule — warm-up uses MAE only, then full composite loss
                       with Lab and frequency terms activated
    4. Param groups updated for v18 module names
       (conditioner, encoder.film, usgs.expert_heads, fusion)
    5. Per-group LR ratios tuned:
       - conditioner (GIV): low LR — it must be stable
       - expert_heads: high LR + high WD — they are the core
       - fusion: medium LR — last-mile refinement
    6. OneCycleLR kept for phase 1, ExponentialLR for phase 2
    7. forward() updated for v18 output dict
    """

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

        # ── Loss components ──────────────────────────────────────────
        self.mae_loss = nn.L1Loss()
        self.logcosh_loss = LogCoshLoss()
        self.grad_loss = GradLoss()
        self.lab_loss = CIELabLoss()  # dE76 proxy (dense smooth guide)
        self.de_loss = DeltaE2000Loss()  # NEW: differentiable exact metric
        self.freq_loss = FrequencyLoss()  # NEW: texture fidelity

        # ── Metrics ──────────────────────────────────────────────────
        self.psnr_metric = PSNR(data_range=1.0)
        self.ssim_metric = SSIM(data_range=1.0)
        self.de_metric = DeltaE()

        # ── Loss weights ─────────────────────────────────────────────
        # Tuned for RYYB→RGB: color fidelity > texture > smoothness
        self.w_de = 1.0    # Differentiable CIEDE2000 — matches the exact
                           # eval metric; primary dE driver.
        self.w_lab = 0.5   # Lab-L1 (dE76 proxy): dense smooth guide, now
                           # secondary to the true dE2000 term above.
        self.w_freq = 0.3  # Frequency loss — raised from 0.1: at 0.1 the
                           # term (~2e-3) was numerically inert. Tunable.
        self.w_grad = 0.5  # Gradient loss weight
        self.w_ssim = 0.2  # SSIM loss weight
        self.w_aux = 0.15  # Auxiliary head supervision weight
        self.w_tv = 0.0    # TV dropped: contribution was ~0 and the run
                           # shows no overfitting, so the regularizer is
                           # unnecessary. Restore a small value if color
                           # speckle appears.

        self.save_hyperparameters(ignore=['model'])

    # ── Initialization ───────────────────────────────────────────────

    def setup(self, stage: str = None) -> None:
        if stage != 'fit' and stage is not None:
            return

        for name, m in self.model.named_modules():

            if isinstance(m, nn.Conv2d):
                # residual-around-identity color matrices start as a no-op
                if 'local_ccm' in name:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    continue
                # Low-gain init for hypernetwork components
                # (prevents parameter explosion at training start)
                hyper_nets = [
                    'expert_heads', 'proj', 'film', 'conditioner', 'chi_net'
                ]
                if any(x in name for x in hyper_nets):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                else:
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_out',
                                            nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # FiLM projections + CCM: zero-init → identity at start
                # (GIV has no effect at epoch 0, grows gradually)
                if 'film' in name or 'conditioner' in name or 'ccm' in name:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # MODEL_PATH = '.experiments/ggpd.hgsa_v18.huawei/logs/checkpoints/_last.ckpt'
        # models.load_model(self.model, 'model', MODEL_PATH)

        Logger.info('HGSA_v18: Pipeline initialized.')

    # ── Optimizer & Scheduler ────────────────────────────────────────

    def configure_optimizers(self):
        """
        Five parameter groups, tuned for v18 module structure:

        conditioner  — GIV extractor. Must be stable → low LR.
                       If it oscillates, all expert conditioning breaks.
        encoder      — Full-res spatial features. Medium LR.
        experts      — USGS manifold core. Highest LR + highest WD.
                       These are the parameters that determine dE.
        film_layers  — FiLM modulation (gamma, beta). Low LR.
                       Zero-initialized, grows slowly by design.
        fusion       — LaplacianGatedFusion + chi_net. Medium LR.
        """
        groups = {
            k: []
            for k in ['conditioner', 'encoder', 'experts', 'film', 'fusion']
        }

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'conditioner' in name or 'ccm' in name:
                groups['conditioner'].append(param)
            elif 'encoder' in name:
                groups['encoder'].append(param)
            elif any(x in name for x in [
                    'expert_heads', 'chi_net', 'spectral_calibrator',
                    'mu_base', 'mu_scale', 'w_init', 'sigma_init'
            ]):
                groups['experts'].append(param)
            elif 'film' in name:
                groups['film'].append(param)
            else:
                groups['fusion'].append(param)

        optimizer = optim.AdamW(
            [
                # GIV conditioner: very stable, low LR
                {
                    'params': groups['conditioner'],
                    'lr': self.lr * 0.5,
                    'weight_decay': self.weight_decay
                },
                # Full-res encoder
                {
                    'params': groups['encoder'],
                    'lr': self.lr * 0.8,
                    'weight_decay': self.weight_decay
                },
                # USGS manifold experts: core of the model
                {
                    'params': groups['experts'],
                    'lr': self.lr * 1.2,
                    'weight_decay': 1e-3
                },
                # FiLM layers: zero-init, grows slowly
                {
                    'params': groups['film'],
                    'lr': self.lr * 0.3,
                    'weight_decay': 0.0
                },
                # Fusion + chi
                {
                    'params': groups['fusion'],
                    'lr': self.lr * 1.0,
                    'weight_decay': self.weight_decay
                },
            ],
            weight_decay=self.weight_decay,
        )

        # LR schedule: a single OneCycle across the ENTIRE run, stepped
        # every optimizer step. Warms up to the per-group peak at 15% of
        # the run, then cosine-anneals down to peak/1000.
        #
        # This replaces the previous two-phase (OneCycle→ExponentialLR)
        # design, which had two problems:
        #   1. configure_optimizers returned only {'optimizer': ...} with
        #      no 'lr_scheduler' key, so Lightning never stepped anything
        #      and the LR stayed frozen for the whole run.
        #   2. ExponentialLR was constructed AFTER OneCycleLR had already
        #      mutated the group LRs, so it captured the wrong base_lrs.
        # A full-run OneCycle is robust and provides the annealing tail
        # that lets the model settle past the observed plateau.
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            # max_lr must match all param groups
            max_lr=[
                self.lr * 0.5,
                self.lr * 0.8,
                self.lr * 1.2,
                self.lr * 0.3,
                self.lr * 1.0,
            ],
            total_steps=total_steps,
            pct_start=0.15,
            div_factor=10,
            final_div_factor=100,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    # ── Losses ───────────────────────────────────────────────────────

    def total_variation_loss(self, img: torch.Tensor) -> torch.Tensor:
        diff_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        diff_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()

    def _compute_loss(
        self,
        main_out: torch.Tensor,
        aux_out: torch.Tensor,
        tgt: torch.Tensor,
        is_warmup: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Composite loss with two phases:

        Warm-up (epoch < warmup_epochs):
            Simple MAE + SSIM + aux supervision.
            Reason: the GIV (conditioner) and FiLM layers are
            zero-initialized. Hitting them with Lab loss before
            they produce meaningful gradients wastes early epochs.

        Main training:
            Full composite loss with Lab + Frequency terms activated.
            Lab loss is the primary driver of Delta-E reduction.
            Frequency loss compensates for Gaussian manifold smoothing.
        """
        main_c = torch.clamp(main_out, 0.0, 1.0)

        loss_mae = self.mae_loss(main_out, tgt)
        loss_ssim = 1.0 - torch.clamp(self.ssim_metric(main_c, tgt), 0., 1.)
        loss_aux = self.mae_loss(aux_out, tgt)

        if is_warmup:
            # Phase 1: simple supervision only
            total = (0.7 * loss_mae + 0.2 * loss_ssim + self.w_aux * loss_aux)
            details = {
                'loss_mae': loss_mae,
                'loss_ssim': loss_ssim,
                'loss_aux': loss_aux,
            }
        else:
            # Phase 2: full composite loss
            loss_lab = self.lab_loss(main_c, tgt)
            loss_de = self.de_loss(main_c, tgt)
            loss_freq = self.freq_loss(main_c, tgt)
            loss_grad = self.grad_loss(main_c, tgt)
            loss_tv = self.total_variation_loss(main_c)

            total = (loss_mae + self.w_de * loss_de +
                     self.w_lab * loss_lab +
                     self.w_freq * loss_freq + self.w_grad * loss_grad +
                     self.w_ssim * loss_ssim + self.w_aux * loss_aux +
                     self.w_tv * loss_tv)
            details = {
                'loss_mae': loss_mae,
                'loss_de2000': loss_de,
                'loss_lab': loss_lab,
                'loss_freq': loss_freq,
                'loss_grad': loss_grad,
                'loss_ssim': loss_ssim,
                'loss_aux': loss_aux,
                'loss_tv': loss_tv,
            }

        return total, details

    # ── Steps ────────────────────────────────────────────────────────

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        src, tgt = batch
        main_out, aux_out = self(src)

        is_warmup = self.current_epoch < self.warmup_epochs
        total_loss, details = self._compute_loss(main_out, aux_out, tgt,
                                                 is_warmup)

        # Log all loss components
        self.log('train_loss', total_loss, prog_bar=True)
        for k, v in details.items():
            self.log(f'train/{k}', v, prog_bar=False)

        # Metrics (no grad needed)
        with torch.no_grad():
            main_c = torch.clamp(main_out, 0.0, 1.0)
            self.log('train_psnr',
                     self.psnr_metric(main_c, tgt),
                     prog_bar=True)
            self.log('train_ssim',
                     self.ssim_metric(main_c, tgt),
                     prog_bar=True)
            self.log('train_de', self.de_metric(main_c, tgt), prog_bar=True)

        return total_loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        src, tgt = batch
        y = torch.clamp(self(src), 0.0, 1.0)

        psnr = self.psnr_metric(y, tgt)
        ssim = self.ssim_metric(y, tgt)
        de = self.de_metric(y, tgt).mean()

        self.log('val_psnr', psnr, prog_bar=True)
        self.log('val_ssim', ssim, prog_bar=True)
        self.log('val_de', de, prog_bar=True)
        # val_loss = dE so ModelCheckpoint can minimize it
        self.log('val_loss', de, prog_bar=True)

        return psnr

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        src, tgt = batch
        y = torch.clamp(self(src), 0.0, 1.0)

        psnr = self.psnr_metric(y, tgt)
        ssim = self.ssim_metric(y, tgt)
        de = self.de_metric(y, tgt).mean()

        self.log('test_psnr', psnr, prog_bar=True)
        self.log('test_ssim', ssim, prog_bar=True)
        self.log('test_de', de, prog_bar=True)
        self.log('test_loss', de, prog_bar=True)

        return psnr

    # ── Inference ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Training: returns (main_out, aux_out) — both raw (no clamp).
        Inference: returns main_out clamped to [0, 1].

        Clamping is intentionally deferred to the step methods
        so that loss functions receive the full gradient signal
        through values slightly outside [0, 1].
        """
        out = self.model(src=x)['res']
        if self.training:
            # v18 returns (final_out, aux_out) during training
            return out
        # Inference: single tensor
        if isinstance(out, (tuple, list)):
            return torch.clamp(out[0], 0.0, 1.0)
        return torch.clamp(out, 0.0, 1.0)
