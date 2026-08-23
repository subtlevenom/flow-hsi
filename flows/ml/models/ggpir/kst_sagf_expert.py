"""KST-SAGF-Expert — sharpness-first expert bottlenecks for 31-band HSI enhancement.

This module reuses the best parts of ``kst_sagf.py`` and reframes them around the
user's new architecture:

  1. a dedicated sharpness / structure stage;
  2. 8 learned expert routes (not explicit channel splits), each with attention
     + conditional bottleneck to a 4-channel latent;
  3. a shared heavy color-matching head operating on the canonical expert latents;
  4. lightweight expert decoders;
  5. pyramid fusion for the final HSI output.

The model keeps the strongest reusable blocks from the KST-SAGF stack:

  * ``DegradationAwareConditioner`` for global degradation context;
  * ``V15CoeffEncoder`` / ``AdvancedGFFN`` style multiscale spectral refinement;
  * ``ChannelMixAttention``-style spectral mixing, repurposed as an expert router;
  * ``PyramidChannelFusion`` for coarse-to-fine final fusion.

Compared to the original KST-SAGF, this version removes the Sprecher/SAGF
transport core and replaces it with explicit expert latent routing designed for
HSI sharpening + spectral/color alignment.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hgsa.hgsa_hsi_v18 import DegradationAwareConditioner
from .bc_usgs import IlluminationEstimator, WaveletUpsampler, _make_msab


class AdvancedGFFN(nn.Module):
    """Gated feed-forward block used throughout the expert model."""

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
    """Spectral mixing stem, repurposed as expert router input.

    It does not hard-split the channels; instead it learns a compact
    representation from the 31-band input that the expert bank can route from.
    """

    def __init__(self, bands: int, dim: int = 64, num_blocks: int = 2, heads: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(bands, dim, 3, padding=1)
        self.body = _make_msab(dim, num_blocks=num_blocks, heads=heads)
        self.head = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(img)))


class PyramidChannelFusion(nn.Module):
    """Lightweight channel-attention fusion over a list of same-resolution maps."""

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
    """Dedicated structural enhancement block before expert routing."""

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
        f1 = F.interpolate(f1, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        f2 = self.enc2(F.avg_pool2d(f0, 4))
        f2 = F.interpolate(f2, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([f0, f1, f2], dim=1))
        detail = self.detail(x)
        feat = self.gffn(self.norm(fused)) + fused + detail
        return feat, illu_map


class ExpertLatentBlock(nn.Module):
    """One learned expert: attention + conditional bottleneck to 4 channels."""

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
                stochastic: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pre(x)
        h = self.attn(h)
        gb = self.cond(cond_vec).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gb.chunk(2, dim=1)
        h = self.norm(h) * (1.0 + torch.tanh(gamma)) + beta
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 4.0)
        if stochastic and self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        z = self.latent(h) + self.gamma * z
        return z, mu, logvar


class SharedColorMatchingHead(nn.Module):
    """Heavy shared head for canonical color / spectral alignment."""

    def __init__(self, latent_dim: int = 4, feat_dim: int = 96, bands: int = 31,
                 num_blocks: int = 4, heads: int = 4):
        super().__init__()
        self.pre = nn.Conv2d(latent_dim * 8 + feat_dim, feat_dim, 1)
        self.body0 = _make_msab(feat_dim, num_blocks=num_blocks, heads=heads)
        self.body1 = _make_msab(feat_dim, num_blocks=num_blocks, heads=heads)
        self.down = nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1)
        self.up = nn.Conv2d(feat_dim, feat_dim, 3, padding=1)
        self.fuse = AdvancedGFFN(feat_dim * 3, feat_dim)
        self.to_bands = nn.Conv2d(feat_dim, bands, 1)

    def forward(self, latents: torch.Tensor, sharp_feat: torch.Tensor) -> torch.Tensor:
        h = self.pre(torch.cat([latents, sharp_feat], dim=1))
        h0 = self.body0(h)
        h1 = self.body1(self.down(h0))
        h1 = F.interpolate(h1, size=h0.shape[-2:], mode='bilinear', align_corners=False)
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
        h = self.stem(z)
        h = self.mix(h)
        h = self.ffn(h)
        return self.out(h)


class ExpertRouter(nn.Module):
    """Eight learned expert routes with implicit specialization and 4-channel bottlenecks."""

    def __init__(self, feat_dim: int = 96, latent_dim: int = 4, cond_dim: int = 96,
                 num_experts: int = 8, heads: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, num_experts, 1),
        )
        self.experts = nn.ModuleList([
            ExpertLatentBlock(feat_dim, latent_dim=latent_dim, cond_dim=cond_dim, heads=heads)
            for _ in range(num_experts)
        ])
        self.latent_fuse = nn.Sequential(
            nn.Conv2d(num_experts * latent_dim, num_experts * latent_dim, 1),
            nn.GELU(),
            nn.Conv2d(num_experts * latent_dim, num_experts * latent_dim, 1),
        )

    def forward(self, feat: torch.Tensor, cond_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.softmax(self.router(feat), dim=1)  # [B, E, H, W]
        latents, mus, logvars = [], [], []
        for i, expert in enumerate(self.experts):
            z, mu, logvar = expert(feat, cond_vec, stochastic=True)
            w = gate[:, i:i + 1]
            latents.append(z * w)
            mus.append(mu)
            logvars.append(logvar)
        z_cat = torch.cat(latents, dim=1)
        z_cat = self.latent_fuse(z_cat)
        mu = torch.cat(mus, dim=1)
        logvar = torch.cat(logvars, dim=1)
        return z_cat, mu, logvar


class KSTSagfExpertLevel(nn.Module):
    """One level of the expert HSI restoration pyramid."""

    def __init__(self, bands: int = 31, feat_dim: int = 96, latent_dim: int = 4,
                 cond_dim: int = 96, illu_dim: int = 32, num_experts: int = 8):
        super().__init__()
        self.bands = bands
        self.giv = DegradationAwareConditioner(bands, cond_dim)
        self.sharp = SharpnessBlock(bands, feat_dim=feat_dim, illu_dim=illu_dim)
        self.stem = ChannelMixAttention(bands, dim=feat_dim, num_blocks=2, heads=4)
        self.router = ExpertRouter(feat_dim=feat_dim, latent_dim=latent_dim,
                                    cond_dim=cond_dim, num_experts=num_experts, heads=4)
        self.shared_head = SharedColorMatchingHead(latent_dim=latent_dim, feat_dim=feat_dim,
                                                   bands=bands, num_blocks=4, heads=4)
        self.decoders = nn.ModuleList([
            ExpertDecoder(latent_dim=latent_dim, bands=bands, feat_dim=48)
            for _ in range(num_experts)
        ])
        self.fusion = PyramidChannelFusion(bands, n_levels=3)
        self.out_gate = nn.Parameter(torch.tensor(0.2))
        self.refine = nn.Sequential(
            nn.Conv2d(bands * 3, bands * 2, 1),
            nn.GroupNorm(2, bands * 2),
            nn.GELU(),
            nn.Conv2d(bands * 2, bands, 3, padding=1),
        )

    def forward(self, src: torch.Tensor) -> dict:
        giv = self.giv(src)
        sharp_feat, illu_map = self.sharp(src)
        stem_feat = self.stem(src)
        feat = sharp_feat + stem_feat
        feat = feat + torch.tanh(illu_map)
        latents, mu, logvar = self.router(feat, giv)
        shared = self.shared_head(latents, feat)
        expert_outs = [dec(latents[:, i * 4:(i + 1) * 4]) for i, dec in enumerate(self.decoders)]
        expert_mix = torch.stack(expert_outs, dim=0).mean(dim=0)
        fused = self.fusion([shared, expert_mix, src])
        y = self.refine(torch.cat([fused, shared, expert_mix], dim=1))
        y = src + self.out_gate * torch.tanh(y)
        return {
            'res': y,
            'shared': shared,
            'expert_mix': expert_mix,
            'latents': latents,
            'mu': mu,
            'logvar': logvar,
            'illu_map': illu_map,
        }


class KSTSagfExpertPyramid(nn.Module):
    """Three-level coarse-to-fine pyramid version of the expert model."""

    def __init__(self, bands: int = 31, feat_dim: int = 96, latent_dim: int = 4,
                 cond_dim: int = 96, illu_dim: int = 32, num_experts: int = 8,
                 upsample: str = 'bilinear'):
        super().__init__()
        assert upsample in ('wavelet', 'bilinear')
        self.upsample = upsample
        self.coarse = KSTSagfExpertLevel(bands, feat_dim, latent_dim, cond_dim, illu_dim, num_experts)
        self.mid = KSTSagfExpertLevel(bands, feat_dim, latent_dim, cond_dim, illu_dim, num_experts)
        self.fine = KSTSagfExpertLevel(bands, feat_dim, latent_dim, cond_dim, illu_dim, num_experts)
        self.fusion = PyramidChannelFusion(bands)
        if upsample == 'wavelet':
            self.up_mid = WaveletUpsampler(bands)
            self.up_fine = WaveletUpsampler(bands)
        else:
            self.up_mid = None
            self.up_fine = None

    def _upsample_to(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, src: torch.Tensor):
        if self.upsample == 'wavelet':
            H, W = src.shape[-2:]
            Hp = (H + 3) // 4 * 4
            Wp = (W + 3) // 4 * 4
            if Hp != H or Wp != W:
                src = F.pad(src, (0, Wp - W, 0, Hp - H), mode='reflect')
            src2 = F.avg_pool2d(src, 2)
            src4 = F.avg_pool2d(src2, 2)
            y4 = self.coarse(src4)['res']
            y2 = self.mid(src2, self.up_mid(y4))['res'] if False else self.mid(src2)['res']
            y = self.fine(src, self.up_fine(y2))['res'] if False else self.fine(src)['res']
        else:
            src2 = F.interpolate(src, scale_factor=0.5, mode='bilinear', align_corners=False)
            src4 = F.interpolate(src, scale_factor=0.25, mode='bilinear', align_corners=False)
            y4 = self.coarse(src4)['res']
            y2 = self.mid(src2)['res']
            y = self.fine(src)['res']

        y2u = self._upsample_to(y2, y.shape[-2:])
        y4u = self._upsample_to(y4, y.shape[-2:])
        y = self.fusion([y, y2u, y4u])
        if self.training:
            return y, y2, y4
        return y


class KSTSagfExpertNet(nn.Module):
    """Single-level expert HSI model built from the KST-SAGF best reusable pieces."""

    def __init__(self, bands: int = 31, feat_dim: int = 96, latent_dim: int = 4,
                 cond_dim: int = 96, illu_dim: int = 32, num_experts: int = 8):
        super().__init__()
        self.level = KSTSagfExpertLevel(bands=bands, feat_dim=feat_dim,
                                        latent_dim=latent_dim, cond_dim=cond_dim,
                                        illu_dim=illu_dim, num_experts=num_experts)

    def forward(self, src: torch.Tensor):
        out = self.level(src)
        if self.training:
            return out
        return out['res']
