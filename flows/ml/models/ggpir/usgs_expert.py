"""USGS-Expert — dual-branch expert bottlenecks for 31-band HSI enhancement.

Reworks the ``kst_sagf`` stack into a mixture-of-experts restorer whose output
is produced by two explicitly separated branches so the training objective can
specialise each one:

  * a **structural / sharpness branch** (``struct``) — an illumination-gated,
    multi-scale MSAB detail enhancer, supervised by SSIM-like structural terms;
  * a **color-matching branch** (``color``) — a heavy shared spectral head over
    the expert latents, supervised by L1 / dE / spectral-angle terms.

The rest of the pipeline is preserved: a ``DegradationAwareConditioner`` for
global degradation context, an 8-route expert bank (soft-routed, each with a
conditional bottleneck to a ``latent_dim``-channel latent), lightweight expert
decoders for local support, and ``PyramidChannelFusion`` coarse-to-fine fusion.
The two branches are combined into the final residual output ``res``.

Fixes over the initial draft: the illumination map is projected to the feature
width before gating (the raw add was a channel mismatch), the shared head and
expert-decoder slicing follow ``num_experts``/``latent_dim`` (no hard-coded 8/4),
and the pyramid actually threads each coarse estimate into the finer level.

Every capacity module is a near-identity, bare-``Parameter``-gated residual so
the network starts close to the identity and is robust to the training
pipeline's generic Conv/Linear re-init.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hgsa.hgsa_hsi_v18 import DegradationAwareConditioner
from .bc_usgs import IlluminationEstimator, WaveletUpsampler, _make_msab


class AdvancedGFFN(nn.Module):
    """Gated feed-forward block (hgsa_v15 ``Advanced_GFFN``) used throughout."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.project_in = nn.Conv2d(in_dim, out_dim * 2, 1)
        self.cal = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim * 2, out_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.dw3 = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim)
        self.dw5 = nn.Conv2d(out_dim, out_dim, 5, padding=2, groups=out_dim)
        self.project_out = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.project_in(x)
        c = c * self.cal(c)
        x1, x2 = c.chunk(2, dim=1)
        return self.project_out(self.dw3(x1) * torch.sigmoid(self.dw5(x2)))


class GatedChannelMix(nn.Module):
    """Lightweight residual MSAB wrapper with a learnable near-identity gate."""

    def __init__(self, ch: int, heads: int = 4, num_blocks: int = 1):
        super().__init__()
        self.mix = _make_msab(ch, num_blocks=num_blocks, heads=heads)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * (self.mix(x) - x)


class ChannelMixAttention(nn.Module):
    """Spectral mixing stem that yields a compact feature the experts route from."""

    def __init__(self, bands: int, dim: int = 64, num_blocks: int = 2,
                 heads: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(bands, dim, 3, padding=1)
        self.body = _make_msab(dim, num_blocks=num_blocks, heads=heads)
        self.head = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(img)))


class PyramidChannelFusion(nn.Module):
    """Lightweight channel-attention fusion over same-resolution band maps."""

    def __init__(self, bands: int, n_levels: int = 3):
        super().__init__()
        c = bands * n_levels
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(1, c // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(1, c // 4), c, 1),
            nn.Sigmoid(),
        )
        self.reduce = nn.Conv2d(c, bands, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(feats, dim=1)
        x = x * self.attn(x)
        return feats[0] + self.gamma * self.reduce(x)


class SharpnessBlock(nn.Module):
    """Illumination-gated multi-scale MSAB detail/structure encoder."""

    def __init__(self, bands: int, feat_dim: int = 96, illu_dim: int = 32):
        super().__init__()
        self.illu = IlluminationEstimator(bands, illu_dim)
        self.stem = nn.Conv2d(bands, feat_dim, 3, padding=1)
        self.illu_gate = nn.Conv2d(illu_dim, feat_dim, 1)
        self.enc0 = _make_msab(feat_dim, num_blocks=2, heads=4)
        self.enc1 = _make_msab(feat_dim, num_blocks=2, heads=4)
        self.enc2 = _make_msab(feat_dim, num_blocks=2, heads=4)
        self.fuse = AdvancedGFFN(feat_dim * 3, feat_dim)
        self.norm = nn.GroupNorm(min(4, feat_dim), feat_dim)
        self.gffn = AdvancedGFFN(feat_dim, feat_dim)
        # Explicit high-frequency detail path (depthwise) added to the features.
        self.detail = nn.Sequential(
            nn.Conv2d(bands, feat_dim, 3, padding=1),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            nn.Conv2d(feat_dim, feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        illu_fea, illu_map = self.illu(x)
        f = self.stem(x) * torch.sigmoid(self.illu_gate(illu_fea))
        f0 = self.enc0(f)
        f1 = self.enc1(F.avg_pool2d(f0, 2))
        f1 = F.interpolate(f1, size=f0.shape[-2:], mode='bilinear',
                           align_corners=False)
        f2 = self.enc2(F.avg_pool2d(f0, 4))
        f2 = F.interpolate(f2, size=f0.shape[-2:], mode='bilinear',
                           align_corners=False)
        fused = self.fuse(torch.cat([f0, f1, f2], dim=1))
        feat = self.gffn(self.norm(fused)) + fused + self.detail(x)
        return feat, illu_map


class ExpertLatentBlock(nn.Module):
    """One learned expert: attention + conditional bottleneck to ``latent_dim``."""

    def __init__(self, feat_dim: int, latent_dim: int = 4, cond_dim: int = 96,
                 heads: int = 4):
        super().__init__()
        self.pre = GatedChannelMix(feat_dim, heads=heads, num_blocks=1)
        self.attn = _make_msab(feat_dim, num_blocks=1, heads=heads)
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )
        self.norm = nn.GroupNorm(min(4, feat_dim), feat_dim)
        self.latent = nn.Conv2d(feat_dim, latent_dim, 1)
        self.mu = nn.Conv2d(feat_dim, latent_dim, 1)
        self.logvar = nn.Conv2d(feat_dim, latent_dim, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor,
                stochastic: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pre(x)
        h = self.attn(h)
        gb = self.cond(cond_vec).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gb.chunk(2, dim=1)
        h = self.norm(h) * (1.0 + torch.tanh(gamma)) + beta
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 4.0)
        if stochastic and self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        z = self.latent(h) + self.gamma * z
        return z, mu, logvar


class ExpertRouter(nn.Module):
    """Soft-routed bank of experts, each with a ``latent_dim`` bottleneck."""

    def __init__(self, feat_dim: int = 96, latent_dim: int = 4,
                 cond_dim: int = 96, num_experts: int = 8, heads: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.router = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, num_experts, 1),
        )
        self.experts = nn.ModuleList([
            ExpertLatentBlock(feat_dim, latent_dim=latent_dim,
                              cond_dim=cond_dim, heads=heads)
            for _ in range(num_experts)
        ])
        lc = num_experts * latent_dim
        self.latent_fuse = nn.Sequential(
            nn.Conv2d(lc, lc, 1), nn.GELU(), nn.Conv2d(lc, lc, 1))

    def forward(self, feat: torch.Tensor, cond_vec: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.softmax(self.router(feat), dim=1)       # [B, E, H, W]
        latents, mus, logvars = [], [], []
        for i, expert in enumerate(self.experts):
            z, mu, logvar = expert(feat, cond_vec, stochastic=True)
            latents.append(z * gate[:, i:i + 1])
            mus.append(mu)
            logvars.append(logvar)
        z_cat = self.latent_fuse(torch.cat(latents, dim=1))
        return z_cat, torch.cat(mus, dim=1), torch.cat(logvars, dim=1)


class SharedColorMatchingHead(nn.Module):
    """Heavy shared head for canonical color / spectral alignment."""

    def __init__(self, latent_dim: int = 4, feat_dim: int = 96, bands: int = 31,
                 num_experts: int = 8, num_blocks: int = 4, heads: int = 4):
        super().__init__()
        self.pre = nn.Conv2d(latent_dim * num_experts + feat_dim, feat_dim, 1)
        self.body0 = _make_msab(feat_dim, num_blocks=num_blocks, heads=heads)
        self.body1 = _make_msab(feat_dim, num_blocks=num_blocks, heads=heads)
        self.down = nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1)
        self.up = nn.Conv2d(feat_dim, feat_dim, 3, padding=1)
        self.fuse = AdvancedGFFN(feat_dim * 3, feat_dim)
        self.to_bands = nn.Conv2d(feat_dim, bands, 1)

    def forward(self, latents: torch.Tensor,
                sharp_feat: torch.Tensor) -> torch.Tensor:
        h = self.pre(torch.cat([latents, sharp_feat], dim=1))
        h0 = self.body0(h)
        h1 = self.body1(self.down(h0))
        h1 = F.interpolate(h1, size=h0.shape[-2:], mode='bilinear',
                           align_corners=False)
        h2 = self.up(h0)
        h = self.fuse(torch.cat([h0, h1, h2], dim=1))
        return self.to_bands(h)


class ExpertDecoder(nn.Module):
    """Lightweight per-expert decoder for local reconstruction support."""

    def __init__(self, latent_dim: int = 4, bands: int = 31, feat_dim: int = 48):
        super().__init__()
        self.stem = nn.Conv2d(latent_dim, feat_dim, 1)
        self.mix = GatedChannelMix(feat_dim, heads=2, num_blocks=1)
        self.ffn = AdvancedGFFN(feat_dim, feat_dim)
        self.out = nn.Conv2d(feat_dim, bands, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.ffn(self.mix(self.stem(z))))


class USGSExpertLevel(nn.Module):
    """One level of the dual-branch expert HSI restoration pyramid."""

    def __init__(self, bands: int = 31, feat_dim: int = 96, latent_dim: int = 4,
                 cond_dim: int = 96, illu_dim: int = 32, num_experts: int = 8):
        super().__init__()
        self.bands = bands
        self.latent_dim = latent_dim
        self.giv = DegradationAwareConditioner(bands, cond_dim)
        self.sharp = SharpnessBlock(bands, feat_dim=feat_dim, illu_dim=illu_dim)
        self.stem = ChannelMixAttention(bands, dim=feat_dim, num_blocks=2, heads=4)
        # Project the (bands-wide) illumination map to the feature width so it
        # can gate the fused feature without a channel mismatch.
        self.illu_gate = nn.Conv2d(bands, feat_dim, 1)
        # --- structural / sharpness branch head (feature -> bands) ---
        self.struct_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, bands, 3, padding=1))
        self.struct_gate = nn.Parameter(torch.tensor(0.2))
        # --- color-matching branch: expert router + shared spectral head ---
        self.router = ExpertRouter(feat_dim=feat_dim, latent_dim=latent_dim,
                                   cond_dim=cond_dim, num_experts=num_experts,
                                   heads=4)
        self.shared_head = SharedColorMatchingHead(
            latent_dim=latent_dim, feat_dim=feat_dim, bands=bands,
            num_experts=num_experts, num_blocks=4, heads=4)
        self.color_gate = nn.Parameter(torch.tensor(0.2))
        self.decoders = nn.ModuleList([
            ExpertDecoder(latent_dim=latent_dim, bands=bands, feat_dim=48)
            for _ in range(num_experts)
        ])
        # --- final fusion of the two branches (+ expert support) ---
        self.fusion = PyramidChannelFusion(bands, n_levels=3)
        self.refine = nn.Sequential(
            nn.Conv2d(bands * 3, bands * 2, 1),
            nn.GroupNorm(2, bands * 2),
            nn.GELU(),
            nn.Conv2d(bands * 2, bands, 3, padding=1))
        self.out_gate = nn.Parameter(torch.tensor(0.2))

    def forward(self, src: torch.Tensor,
                coarse_out: torch.Tensor = None) -> dict:
        x_in = src if coarse_out is None else src + coarse_out
        giv = self.giv(x_in)                                 # [B, cond_dim]
        sharp_feat, illu_map = self.sharp(x_in)              # feat, [B,bands,H,W]
        feat = sharp_feat + self.stem(x_in)
        feat = feat * torch.sigmoid(self.illu_gate(illu_map))

        # Structural branch — residual detail/structure enhancement.
        struct = x_in + self.struct_gate * torch.tanh(self.struct_head(sharp_feat))

        # Color branch — expert latents -> shared spectral alignment head.
        latents, mu, logvar = self.router(feat, giv)
        color = x_in + self.color_gate * torch.tanh(self.shared_head(latents, feat))

        # Per-expert local support (mean of lightweight decoders).
        d = self.latent_dim
        expert_outs = [dec(latents[:, i * d:(i + 1) * d])
                       for i, dec in enumerate(self.decoders)]
        expert_mix = torch.stack(expert_outs, dim=0).mean(dim=0)

        fused = self.fusion([color, struct, expert_mix])
        y = self.refine(torch.cat([fused, struct, color], dim=1))
        y = x_in + self.out_gate * torch.tanh(y)
        return {'res': y, 'struct': struct, 'color': color,
                'mu': mu, 'logvar': logvar, 'illu_map': illu_map}


class USGSExpertPyramid(nn.Module):
    """Three-level coarse-to-fine dual-branch expert HSI model."""

    def __init__(self, bands: int = 31, feat_dim: int = 96, latent_dim: int = 4,
                 cond_dim: int = 96, illu_dim: int = 32, num_experts: int = 8,
                 upsample: str = 'bilinear'):
        super().__init__()
        assert upsample in ('wavelet', 'bilinear')
        self.upsample = upsample

        def _level() -> USGSExpertLevel:
            return USGSExpertLevel(bands, feat_dim, latent_dim, cond_dim,
                                   illu_dim, num_experts)

        self.coarse = _level()
        self.mid = _level()
        self.fine = _level()
        self.fusion = PyramidChannelFusion(bands)
        if upsample == 'wavelet':
            self.up_mid = WaveletUpsampler(bands)
            self.up_fine = WaveletUpsampler(bands)
        else:
            self.up_mid = None
            self.up_fine = None

    @staticmethod
    def _up(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, src: torch.Tensor):
        if self.upsample == 'wavelet':
            H, W = src.shape[-2:]
            Hp, Wp = (H + 3) // 4 * 4, (W + 3) // 4 * 4
            if Hp != H or Wp != W:
                src = F.pad(src, (0, Wp - W, 0, Hp - H), mode='reflect')
            src2 = F.avg_pool2d(src, 2)
            src4 = F.avg_pool2d(src2, 2)
            y4 = self.coarse(src4)['res']
            y2 = self.mid(src2, coarse_out=self.up_mid(y4))['res']
            fine = self.fine(src, coarse_out=self.up_fine(y2))
        else:
            src2 = F.interpolate(src, scale_factor=0.5, mode='bilinear',
                                 align_corners=False)
            src4 = F.interpolate(src, scale_factor=0.25, mode='bilinear',
                                 align_corners=False)
            y4 = self.coarse(src4)['res']
            y2 = self.mid(src2, coarse_out=self._up(y4, src2.shape[-2:]))['res']
            fine = self.fine(src, coarse_out=self._up(y2, src.shape[-2:]))
            Hp = Wp = None

        y = fine['res']
        y = self.fusion([y, self._up(y2, y.shape[-2:]), self._up(y4, y.shape[-2:])])

        struct, color = fine['struct'], fine['color']
        if self.upsample == 'wavelet' and (Hp != H or Wp != W):
            y = y[..., :H, :W]
            struct = struct[..., :H, :W]
            color = color[..., :H, :W]

        if self.training:
            return {'main': y, 'aux': [y2, y4], 'struct': struct,
                    'color': color, 'mu': fine['mu'], 'logvar': fine['logvar']}
        return y


class USGSExpertNet(nn.Module):
    """Single-level dual-branch expert HSI model."""

    def __init__(self, bands: int = 31, feat_dim: int = 96, latent_dim: int = 4,
                 cond_dim: int = 96, illu_dim: int = 32, num_experts: int = 8):
        super().__init__()
        self.level = USGSExpertLevel(bands=bands, feat_dim=feat_dim,
                                     latent_dim=latent_dim, cond_dim=cond_dim,
                                     illu_dim=illu_dim, num_experts=num_experts)

    def forward(self, src: torch.Tensor):
        out = self.level(src)
        if self.training:
            return {'main': out['res'], 'aux': [], 'struct': out['struct'],
                    'color': out['color'], 'mu': out['mu'],
                    'logvar': out['logvar']}
        return out['res']
