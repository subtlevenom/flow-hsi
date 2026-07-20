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
    """Прямая оптимизация Delta-E в пространстве CIELAB.

    Каналы a*/b* (цветность) взвешены сильнее L* (яркость), т.к. Delta-E
    в первую очередь определяется цветовой ошибкой (chroma_weight).
    """
    _XYZ_REF = torch.tensor([0.95047, 1.00000, 1.08883])
    _RGB2XYZ = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    def __init__(self, chroma_weight: float = 1.5):
        super().__init__()
        # [L*, a*, b*] — усиливаем вклад цветовых каналов a* и b*.
        self.register_buffer(
            '_ch_weights',
            torch.tensor([1.0, chroma_weight, chroma_weight]).view(1, 3, 1, 1)
        )

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
        w = self._ch_weights.to(pred_lab.dtype)
        loss = (w * (pred_lab - target_lab).abs()).mean()
        return loss.to(pred.dtype)

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

class DeltaE2000Loss(nn.Module):
    """Дифференцируемая функция потерь Delta-E CIEDE2000.

    Напрямую оптимизирует тестовую метрику dE (CIE 2000). Все операции
    sqrt/hypot защищены малым eps для устойчивости градиента (особенно
    при идеальном совпадении, где аргумент финального sqrt стремится к 0).
    """
    # sRGB (D65) -> linear -> XYZ (совпадает с CIELabLoss / метрикой rgb_to_lab)
    _XYZ_REF = torch.tensor([0.95047, 1.00000, 1.08883])
    _RGB2XYZ = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb_lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        m = self._RGB2XYZ.to(rgb.device, rgb.dtype)
        xyz = torch.einsum('oi, bihw -> bohw', m, rgb_lin)
        ref = self._XYZ_REF.to(rgb.device, rgb.dtype).view(1, 3, 1, 1)
        xyz_n = xyz / ref
        thr = 0.008856
        f = torch.where(xyz_n > thr, xyz_n.clamp(min=thr) ** (1.0 / 3.0), 7.787 * xyz_n + 4.0 / 29.0)
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return torch.stack([L, a, b], dim=1)

    def _ciede2000(self, lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        deg = 180.0 / math.pi
        L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
        L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

        C1 = torch.sqrt(a1 ** 2 + b1 ** 2 + eps)
        C2 = torch.sqrt(a2 ** 2 + b2 ** 2 + eps)
        C_bar = (C1 + C2) / 2.0
        C_bar7 = C_bar ** 7
        G = 0.5 * (1.0 - torch.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7) + eps))

        a1p = (1.0 + G) * a1
        a2p = (1.0 + G) * a2
        C1p = torch.sqrt(a1p ** 2 + b1 ** 2 + eps)
        C2p = torch.sqrt(a2p ** 2 + b2 ** 2 + eps)

        h1p = (torch.atan2(b1, a1p) * deg) % 360.0
        h2p = (torch.atan2(b2, a2p) * deg) % 360.0

        dLp = L2 - L1
        dCp = C2p - C1p

        dhp = h2p - h1p
        dhp = torch.where(dhp > 180.0, dhp - 360.0, dhp)
        dhp = torch.where(dhp < -180.0, dhp + 360.0, dhp)
        dHp = 2.0 * torch.sqrt(C1p * C2p + eps) * torch.sin(torch.deg2rad(dhp / 2.0))

        Lp_bar = (L1 + L2) / 2.0
        Cp_bar = (C1p + C2p) / 2.0

        hsum = h1p + h2p
        hdiff = torch.abs(h1p - h2p)
        hp_bar = torch.where(
            hdiff > 180.0,
            torch.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0),
            hsum / 2.0,
        )

        T = (1.0
             - 0.17 * torch.cos(torch.deg2rad(hp_bar - 30.0))
             + 0.24 * torch.cos(torch.deg2rad(2.0 * hp_bar))
             + 0.32 * torch.cos(torch.deg2rad(3.0 * hp_bar + 6.0))
             - 0.20 * torch.cos(torch.deg2rad(4.0 * hp_bar - 63.0)))

        d_theta = 30.0 * torch.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
        Cp_bar7 = Cp_bar ** 7
        R_C = 2.0 * torch.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7) + eps)
        Lp_bar_2 = (Lp_bar - 50.0) ** 2
        S_L = 1.0 + (0.015 * Lp_bar_2) / torch.sqrt(20.0 + Lp_bar_2 + eps)
        S_C = 1.0 + 0.045 * Cp_bar
        S_H = 1.0 + 0.015 * Cp_bar * T
        R_T = -torch.sin(torch.deg2rad(2.0 * d_theta)) * R_C

        term_L = dLp / S_L
        term_C = dCp / S_C
        term_H = dHp / S_H
        d_sq = term_L ** 2 + term_C ** 2 + term_H ** 2 + R_T * term_C * term_H
        return torch.sqrt(d_sq.clamp(min=0.0) + eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_32 = pred.clamp(0, 1).to(torch.float32)
        t_32 = target.clamp(0, 1).to(torch.float32)
        lab_p = self._rgb_to_lab(p_32)
        lab_t = self._rgb_to_lab(t_32)
        de = self._ciede2000(lab_p, lab_t).mean()
        return de.to(pred.dtype)

# ─────────────────────────────────────────────────────────────────────
# PIPELINE v22 (Optimized for Volga2K & Hyper-KAN)
# ─────────────────────────────────────────────────────────────────────

class GEOTPipeline_v22(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-3,
                 warmup_epochs: int = 50,
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
        self.lab_loss = CIELabLoss(chroma_weight=1.5)
        self.freq_loss = FrequencyLoss(loss_weight=0.5) # Увеличен вес для Volga2K
        self.grad_loss = GradLoss()
        self.mk_loss = MongeKantorovichLoss()
        self.de_loss = DeltaE2000Loss()           # Дифференцируемый CIEDE2000
        self.ssim_loss = SSIM(data_range=1.0)      # SSIM как функция потерь (1 - ssim)

        # Weights (v22 Volga Optimized)
        self.w_lab = 3.0   # Критично для восстановления цветов в Volga2K
        self.w_freq = 1.2
        self.w_mk = 0.08   # Усиленная регуляризация для стабильности KAN
        self.w_ssim = 0.25
        self.w_de = 0.5    # Прямая оптимизация тестовой метрики dE (CIEDE2000)
        self.w_aux = 0.6   # Повышенное внимание к транспортному выходу
        self.w_grad = 0.5

        self.save_hyperparameters(ignore=['model'])

    @property
    def model(self):
        return self._model.layers.hgsa

    def setup(self, stage: str = None) -> None:
        if stage != 'fit': return
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                # Инициализация KAN-базисов и гиперсетей
                if 'hyper' in name or 'phi_net' in name or 'base' in name or 'spline' in name:
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                # FiLM.proj намеренно стартует с тождественной модуляции (нули).
                # Остальные Linear (MLP кондиционера) инициализируем xavier,
                # иначе giv == 0 и градиент к кондиционеру не течёт (мёртвая ветвь).
                if 'film' in name:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None: nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        groups = {'cond': [], 'enc': [], 'eot': [], 'fuse': []}
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if 'conditioner' in name: groups['cond'].append(param)
            elif 'encoder' in name: groups['enc'].append(param)
            elif 'eot_usgs' in name: groups['eot'].append(param) # Hyper-KAN блоки
            else: groups['fuse'].append(param)

        optimizer = optim.AdamW([
            {'params': groups['cond'], 'lr': self.lr * 0.5},
            {'params': groups['enc'], 'lr': self.lr * 0.8},
            {'params': groups['eot'], 'lr': self.lr * 1.8, 'weight_decay': 1e-3}, # Высокий LR для KAN
            {'params': groups['fuse'], 'lr': self.lr * 1.0},
        ], weight_decay=self.weight_decay)

        steps = self.trainer.estimated_stepping_batches
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.lr * 0.5, self.lr * 0.8, self.lr * 1.8, self.lr * 1.0],
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
        loss_mae = self.mae_loss(main_out, tgt)
        loss_aux = self.mae_loss(aux_out, tgt)

        if is_warmup:
            return loss_mae + self.w_aux * loss_aux, {'mae': loss_mae, 'aux': loss_aux}

        m_c, a_c = torch.clamp(main_out, 0, 1), torch.clamp(aux_out, 0, 1)
        
        loss_lab = self.lab_loss(m_c, tgt)
        loss_freq = self.freq_loss(m_c, tgt)
        loss_grad = self.grad_loss(m_c, tgt)
        loss_mk = self.mk_loss(src, a_c)
        loss_de = self.de_loss(m_c, tgt)
        loss_ssim = 1.0 - self.ssim_loss(m_c, tgt)

        total = (loss_mae + self.w_lab * loss_lab + self.w_freq * loss_freq +
                 self.w_grad * loss_grad + self.w_aux * loss_aux + self.w_mk * loss_mk +
                 self.w_de * loss_de + self.w_ssim * loss_ssim)

        return total, {'mae': loss_mae, 'lab': loss_lab, 'mk': loss_mk,
                       'de': loss_de, 'ssim': loss_ssim}

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
        self.log('val_psnr', psnr, prog_bar=True)
        self.log('val_ssim', ssim, prog_bar=True)
        self.log('val_de', de, prog_bar=True)

        return loss_mae

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)

        loss_mae = F.l1_loss(y, tgt)
        psnr = self.psnr_metric(y, tgt)
        ssim = self.ssim_metric(y, tgt)
        de = self.de_metric(y, tgt).mean()

        self.log('test_mae', loss_mae, prog_bar=True, sync_dist=True)
        self.log('test_psnr', psnr, prog_bar=True)
        self.log('test_ssim', ssim, prog_bar=True)
        self.log('test_de', de, prog_bar=True)

        return loss_mae