from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ggpir_msab_encoder import GGPIRMSABEncoder
from .ggpir_msab_decoder import GGPIRMSABDecoder
from ..hgsa import HGSA


class RawContextCascade(nn.Module):
    """Path-decoupled coefficient context (approach A).

    Builds a Laplacian cascade directly on the *raw* HSI bands (no encoder
    decorrelation) and summarizes each band with mean+variance statistics
    into a single global context vector (``giv_raw``). This vector is
    broadcast to every triplet and FiLM-fused into the Sprecher/USGS
    coefficient head, so the coefficient prediction sees the full, joint,
    multi-scale spectrum while the value/transform path keeps operating on
    the well-conditioned decorrelated triplets.
    """

    def __init__(self, in_channels: int = 31, giv_dim: int = 64,
                 levels: int = 3):
        super().__init__()
        self.levels = levels
        stat_dim = in_channels * 2 * levels          # mean+var per band per level
        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, giv_dim * 2),
            nn.GELU(),
            nn.Linear(giv_dim * 2, giv_dim),
            nn.LayerNorm(giv_dim),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        cur = src
        feats = []
        for lvl in range(self.levels):
            if lvl < self.levels - 1:
                down = F.avg_pool2d(cur, 2)
                up = F.interpolate(down, size=cur.shape[-2:],
                                   mode='bilinear', align_corners=False)
                band = cur - up                       # Laplacian band
                cur = down
            else:
                band = cur                            # coarsest residual
            mu = band.mean(dim=(2, 3))                # [B, C]
            var = band.var(dim=(2, 3))                # [B, C]
            feats.append(torch.cat([mu, var], dim=1))
        stats = torch.cat(feats, dim=1)               # [B, C*2*levels]
        return self.mlp(stats)                        # [B, giv_dim]


class USGSLevel(nn.Module):
    """One coarse-to-fine pyramid level.

    ``decorrelate (MSAB) -> per-triplet Sprecher/USGS -> recompose (MSAB)``,
    with an input residual and a raw-HSI coefficient-context branch. The
    Sprecher/USGS core is used unchanged; only its coefficient GIV is
    augmented with the raw-band context.
    """

    def __init__(self, bands: int = 31, triplet: int = 3,
                 extra_blocks: int = 0, Q: int = 8, M: int = 4,
                 giv_dim: int = 64):
        super().__init__()
        self.bands = bands
        self.encoder = GGPIRMSABEncoder(bands, triplet, extra_blocks=extra_blocks)
        self.usgs = HGSA(triplet, triplet, Q=Q, M=M)
        self.decoder = GGPIRMSABDecoder(triplet, bands)
        self.raw_ctx = RawContextCascade(bands, giv_dim)

    def forward(self, src: torch.Tensor,
                coarse_out: torch.Tensor = None) -> torch.Tensor:
        # Coarse-to-fine: refine this scale's input with the upsampled
        # spectral estimate from the level below.
        x = src
        if coarse_out is not None:
            x = x + F.interpolate(coarse_out, size=src.shape[-2:],
                                  mode='bilinear', align_corners=False)

        # Raw-HSI coefficient context, broadcast across the 31 triplets so it
        # aligns with the encoder's `(b n) c h w` batch-folding (approach A).
        giv_raw = self.raw_ctx(x)                         # [B, giv_dim]
        giv_raw = giv_raw.repeat_interleave(self.bands, dim=0)  # [B*bands, .]

        t = self.encoder(x)                               # [B*bands, 3, H, W]
        y = self.usgs(t, giv_raw=giv_raw)                 # Sprecher/USGS core
        out = self.decoder(y) + x                         # recompose + residual
        return out


class USGSPyramid(nn.Module):
    """Sprecher-USGS Laplacian pyramid.

    Three coarse-to-fine levels (X/4 -> X/2 -> X), each a full
    ``encoder -> USGS -> decoder`` stack with growing depth. Coarse spectral
    estimates are upsampled and added as residuals into finer levels. In
    training the per-scale outputs are returned for deep supervision; at
    inference only the full-resolution output is returned (single tensor,
    so it fits the Flow graph's `res` output).
    """

    def __init__(self, bands: int = 31, triplet: int = 3,
                 depths: List[int] = [1, 3, 5], Q: int = 8, M: int = 4,
                 giv_dim: int = 64):
        super().__init__()
        self.coarse = USGSLevel(bands, triplet, depths[0], Q, M, giv_dim)
        self.mid = USGSLevel(bands, triplet, depths[1], Q, M, giv_dim)
        self.fine = USGSLevel(bands, triplet, depths[2], Q, M, giv_dim)

    def forward(self, src: torch.Tensor):
        x2 = F.interpolate(src, scale_factor=0.5,
                           mode='bilinear', align_corners=False)
        x4 = F.interpolate(src, scale_factor=0.25,
                           mode='bilinear', align_corners=False)

        y4 = self.coarse(x4)
        y2 = self.mid(x2, coarse_out=y4)
        y = self.fine(src, coarse_out=y2)

        if self.training:
            return y, y2, y4          # deep supervision (fine, mid, coarse)
        return y
