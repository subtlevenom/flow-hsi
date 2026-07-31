"""Banded-Corrector USGS (BC-USGS) — Tier A.

A memory-lean, higher-fidelity replacement for the encoder->triplet->decoder
Sprecher/USGS design. Instead of fabricating 31 abstract triplets (which
inflates the value dimensionality 31 -> 93 and folds 31 into the batch), the
Gaussian mixture is applied **directly, per band**:

    value path   : x[B,31,H,W]  --MSAB whitening-->  V[B,31,H,W]   (31 per-band args)
    coeff path   : x[B,31,H,W]  --MSAB + Fourier-->  F_coef[B,D,H,W]
    per-band core: psi_k = sum_q w_kq * exp(-1/2 ((V_k - mu_kq)/sigma_kq)^2)
    spectral read: psi[B,31,Q,H,W] --Q-collapse + MSAB mixing + CCM--> y[B,31,H,W]

New SOTA blocks vs. the current solution:
  * GlobalFourierMixer  — GFNet/SFNet-style global spatial mixing (fp32 FFT)
                          gives the coefficient head a global receptive field
                          for illumination / white-balance context.
  * Wavelength-conditioned param head — smooth-across-band Gaussian params
                          via a learned per-band embedding (spectral-INR
                          flavored) plus per-pixel spatial offsets.
  * Spectral read-out   — MSAB spectral mixing re-couples the bands after the
                          per-band (independent) Gaussian nonlinearity.
  * Global + low-rank local color matrices — explicit linear cross-band
                          correction, residual-around-identity.

Kept from the proven design: the USGS Gaussian core math, MSAB spectral
attention, evenly-spaced mu-grid init, illumination-driven sigma boost,
gradient checkpointing of the M-expert loop, per-level (Q, M), coarse-to-fine
pyramid with deep supervision. The coarse-to-fine recomposition is a
critically-sampled inverse Haar wavelet synthesis (exact, no bilinear).
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flows.ml.layers.mst import MSAB
from flows.ml.layers.mw_isp import DWTInverse
from ..hgsa.hgsa_hsi_v18 import (
    DegradationAwareConditioner,
    GlobalColorMatrix,
    FiLM,
)


def _make_msab(dim: int, num_blocks: int, heads: int) -> MSAB:
    """Build an MSAB with the shape constraint dim_head * heads == dim.

    MS_MSA re-uses the value projection (size dim_head*heads) as the input to
    a `dim`-channel positional conv, so the product must equal `dim`.
    """
    heads = max(1, heads)
    while dim % heads != 0:
        heads -= 1
    dim_head = dim // heads
    return MSAB(dim=dim, dim_head=dim_head, heads=heads, num_blocks=num_blocks)


class GlobalFourierMixer(nn.Module):
    """Global spatial mixing in the Fourier domain (GFNet/SFNet lineage).

    Provides an image-wide receptive field at O(H*W*log(H*W)) so the
    coefficient head can see global illumination / white-balance context when
    predicting the per-pixel Gaussian parameters. The FFT is run in fp32 with
    autocast disabled (rfft2 is unstable / unsupported under bf16-mixed).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proc = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1),
        )
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            xf = torch.fft.rfft2(x32, norm='ortho')          # [B, C, H, Wf] complex
            feat = torch.cat([xf.real, xf.imag], dim=1)       # [B, 2C, H, Wf]
            feat = self.proc(feat)
            re, im = feat.chunk(2, dim=1)
            xf = torch.complex(re, im)
            y = torch.fft.irfft2(xf, s=(H, W), norm='ortho')  # [B, C, H, W]
        return x + self.out(y.to(x.dtype))


class BCValueEncoder(nn.Module):
    """Whitened, per-band value path (stays 31-channel).

    Keeps the one genuinely good idea of the current design — MSAB spectral
    decorrelation for well-conditioned Gaussian arguments — but without the
    31->3->31 fabrication. Output is a residual around the (normalized) raw
    bands, so the arguments stay near the input range that the mu-grid init
    ([0.1, 0.9]) expects.
    """

    def __init__(self, bands: int, dim: int, num_blocks: int = 2,
                 heads: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(bands, dim, 3, padding=1)
        self.body = _make_msab(dim, num_blocks=num_blocks, heads=heads)
        self.head = nn.Conv2d(dim, bands, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body(self.stem(x))
        return x + self.head(f)                               # [B, bands, H, W]


class BCCoeffEncoder(nn.Module):
    """Spatial hypernetwork context for the Gaussian coefficients.

    MSAB (spectral context) -> optional Global Fourier mixer (global spatial
    context) -> FiLM injection of a global scene descriptor (GIV). Produces
    the feature map from which the per-band Gaussian parameters are read.
    """

    def __init__(self, bands: int, dim: int, giv_dim: int,
                 num_blocks: int = 2, heads: int = 4,
                 use_fourier: bool = True):
        super().__init__()
        self.stem = nn.Conv2d(bands, dim, 3, padding=1)
        self.body = _make_msab(dim, num_blocks=num_blocks, heads=heads)
        self.fourier = GlobalFourierMixer(dim) if use_fourier else None
        self.film = FiLM(giv_dim, dim)
        self.norm = nn.GroupNorm(min(4, dim), dim)

    def forward(self, x: torch.Tensor, giv: torch.Tensor) -> torch.Tensor:
        f = self.body(self.stem(x))
        if self.fourier is not None:
            f = self.fourier(f)
        f = self.film(f, giv)
        return self.norm(f)                                   # [B, dim, H, W]


class PerBandUSGS(nn.Module):
    """Per-band Gaussian-mixture core (Tier A).

    For every output band ``k`` and pixel ``(h, w)`` a 1-D Gaussian mixture is
    evaluated on that band's whitened value ``V_k``::

        psi_k = sum_q w_kq * exp(-1/2 ((V_k - mu_kq) / sigma_kq)^2)

    summed over ``M`` experts. The Gaussian parameters are per-band, per-pixel
    offsets (from the coeff features) around learnable priors: an evenly-spaced
    mu grid, plus — for ``param_head='wavelength'`` — a smooth, per-band,
    spatially-constant baseline predicted from a learned band embedding.

    The M-expert loop is gradient-checkpointed: only one expert's parameter
    map ``[B, bands*3*Q, H, W]`` is live at a time.
    """

    def __init__(self, bands: int, Q: int, M: int, dim: int,
                 param_head: str = 'wavelength',
                 emb_dim: int = 16, use_checkpoint: bool = True):
        super().__init__()
        assert param_head in ('wavelength', 'flat')
        self.bands, self.Q, self.M = bands, Q, M
        self.param_head = param_head
        self.use_checkpoint = use_checkpoint

        # Per-pixel spatial offset predictor, one Conv per expert. Kept as a
        # ModuleList so each expert's params can be recomputed independently
        # inside its own checkpoint (memory: one expert live at a time).
        self.expert_param = nn.ModuleList([
            nn.Conv2d(dim, bands * 3 * Q, 1) for _ in range(M)
        ])
        # Small offset scale, robust to the pipeline's generic re-init (a bare
        # Parameter is not touched by the Conv/Linear/BN re-init in setup()).
        self.offset_scale = nn.Parameter(torch.tensor(0.1))

        # Learnable priors (broadcast over batch / space).
        mu_grid = torch.linspace(0.1, 0.9, Q).view(1, 1, Q, 1, 1)
        self.mu_base = nn.Parameter(mu_grid.expand(1, bands, Q, 1, 1).clone())
        self.mu_scale = nn.Parameter(torch.ones(1, bands, Q, 1, 1) * 4.5)
        self.w_init = nn.Parameter(0.1 * torch.randn(1, bands, Q, 1, 1))
        self.sigma_init = nn.Parameter(torch.ones(M, bands, Q) * 0.2)

        # Wavelength-conditioned baseline: a smooth, per-band, spatially
        # constant offset for {w, mu, sigma} x Q, from a learned band
        # embedding.
        if param_head == 'wavelength':
            self.band_emb = nn.Parameter(0.02 * torch.randn(bands, emb_dim))
            self.band_mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2), nn.GELU(),
                nn.Linear(emb_dim * 2, 3 * Q),
            )
        else:
            self.band_emb = None
            self.band_mlp = None

    def _wavelength_bias(self) -> torch.Tensor:
        # [1, bands, 3, Q, 1, 1] additive baseline (spatially constant).
        b = self.band_mlp(self.band_emb)                     # [bands, 3*Q]
        return b.view(1, self.bands, 3, self.Q, 1, 1)

    def _expert_contrib(self, i: int, feat: torch.Tensor,
                        xi: torch.Tensor, sigma_boost: torch.Tensor,
                        wl_bias: torch.Tensor) -> torch.Tensor:
        B = feat.size(0)
        H, W = feat.shape[-2:]
        p_e = self.expert_param[i](feat).view(
            B, self.bands, 3, self.Q, H, W)
        if wl_bias is not None:
            p_e = p_e + wl_bias
        p_e = self.offset_scale * p_e

        w = self.w_init + p_e[:, :, 0]                        # [B, bands, Q, H, W]
        mu = (torch.tanh(self.mu_base + p_e[:, :, 1])
              * F.softplus(self.mu_scale) + 0.5)
        s_b = self.sigma_init[i].view(1, self.bands, self.Q, 1, 1)
        sigma = (F.softplus(s_b + p_e[:, :, 2]) + 0.01) * sigma_boost

        g = torch.exp(-0.5 * ((xi - mu) / sigma).pow(2))      # [B, bands, Q, H, W]
        return w * g

    def forward(self, V: torch.Tensor, feat: torch.Tensor,
                brightness: torch.Tensor) -> torch.Tensor:
        B, bands, H, W = V.shape
        xi = V.unsqueeze(2)                                   # [B, bands, 1, H, W]
        # Illumination-driven sigma boost (cheap, from input brightness).
        sigma_boost = torch.clamp(
            1.0 / (brightness + 1e-4), 1.0, 2.5).unsqueeze(2)  # [B, 1, 1, H, W]

        wl_bias = (self._wavelength_bias()
                   if self.band_mlp is not None else None)

        psi_total = torch.zeros(B, bands, self.Q, H, W,
                                device=V.device, dtype=V.dtype)
        for i in range(self.M):
            if self.use_checkpoint and self.training:
                contrib = checkpoint(
                    self._expert_contrib, i, feat, xi, sigma_boost, wl_bias,
                    use_reentrant=False)
            else:
                contrib = self._expert_contrib(
                    i, feat, xi, sigma_boost, wl_bias)
            psi_total = psi_total + contrib
        return psi_total                                      # [B, bands, Q, H, W]


class LocalColorMatrixLR(nn.Module):
    """Per-pixel low-rank cross-band color matrix, residual around identity.

    A full 31x31 per-pixel matrix (961 channels) is too heavy, so the
    correction is factored as ``U @ (V^T x)`` with rank ``r``. Scaled by a
    small learnable gamma so it starts as a near-no-op and is robust to the
    pipeline's generic weight re-init.
    """

    def __init__(self, feat_dim: int, bands: int, rank: int = 4):
        super().__init__()
        self.bands, self.rank = bands, rank
        self.u = nn.Conv2d(feat_dim, bands * rank, 1)
        self.v = nn.Conv2d(feat_dim, bands * rank, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        U = self.u(feat).view(B, self.bands, self.rank, H, W)   # o, k
        Vt = self.v(feat).view(B, self.rank, self.bands, H, W)  # k, c
        Vx = torch.einsum('bkchw,bchw->bkhw', Vt, x)            # [B, r, H, W]
        delta = torch.einsum('bokhw,bkhw->bohw', U, Vx)        # [B, bands, H, W]
        return x + self.gamma * delta


class SpectralReadout(nn.Module):
    """Collapse the Q axis and re-couple bands after the per-band Gaussians.

    1. SE-style spectral calibration over the (bands*Q) responses.
    2. Grouped Q-collapse (per band, Q -> 1).
    3. MSAB spectral mixing across bands (restores cross-band correlation that
       the per-band Gaussian core cannot represent).
    """

    def __init__(self, bands: int, Q: int, readout: str = 'msab'):
        super().__init__()
        self.bands, self.Q = bands, Q
        sq = bands * Q
        self.calib = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(sq, max(1, sq // 4), 1), nn.GELU(),
            nn.Conv2d(max(1, sq // 4), sq, 1), nn.Sigmoid())
        # groups=bands: each band's Q responses -> a single band value.
        self.q_collapse = nn.Conv2d(sq, bands, 1, groups=bands)
        self.mix = (_make_msab(bands, num_blocks=1, heads=1)
                    if readout == 'msab' else None)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        B, bands, Q, H, W = psi.shape
        psi_flat = psi.reshape(B, bands * Q, H, W)
        psi_flat = psi_flat * self.calib(psi_flat)
        y = self.q_collapse(psi_flat)                        # [B, bands, H, W]
        if self.mix is not None:
            y = self.mix(y)
        return y


class WaveletUpsampler(nn.Module):
    """Exact 2x upsampling via inverse Haar DWT with predicted detail.

    Replaces bilinear upsampling in the coarse-to-fine pyramid. The coarse
    image fills the LL slot (scaled by 2 to preserve magnitude: iDWT of a pure
    LL yields a 0.5x box, so the 2x cancels it); the three high subbands
    (LH, HL, HH) are predicted from the coarse image, so the upsampling injects
    learned high-frequency detail and recomposes exactly through DWTInverse. A
    small gamma keeps it near a plain box-upsample at init (and robust to the
    pipeline's generic conv re-init).
    """

    def __init__(self, bands: int, hidden: int = None):
        super().__init__()
        self.bands = bands
        hidden = hidden or bands * 2
        self.idwt = DWTInverse()
        self.detail = nn.Sequential(
            nn.Conv2d(bands, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, bands * 3, 3, padding=1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        B, C, h, w = ic.shape
        detail = self.detail(ic).view(B, C, 3, h, w)          # LH, HL, HH
        ll = (2.0 * ic).unsqueeze(2)                          # magnitude-preserving LL
        merged = torch.cat([ll, self.gamma * detail], dim=2)  # [B, C, 4, h, w]
        merged = merged.reshape(B, C * 4, h, w)               # band-major (ll,lh,hl,hh)
        return self.idwt(merged)                              # [B, C, 2h, 2w]


class BCUSGSLevel(nn.Module):
    """One coarse-to-fine pyramid level of BC-USGS (Tier A)."""

    def __init__(self, bands: int = 31, Q: int = 8, M: int = 4,
                 dim: int = 64, giv_dim: int = 64, blocks: int = 2,
                 use_fourier: bool = True, param_head: str = 'wavelength',
                 readout: str = 'msab', use_checkpoint: bool = True):
        super().__init__()
        self.bands = bands
        self.giv = DegradationAwareConditioner(bands, giv_dim)
        self.value_enc = BCValueEncoder(bands, dim, num_blocks=2)
        self.coeff_enc = BCCoeffEncoder(
            bands, dim, giv_dim, num_blocks=max(1, blocks),
            use_fourier=use_fourier)
        self.core = PerBandUSGS(
            bands, Q, M, dim, param_head=param_head,
            use_checkpoint=use_checkpoint)
        self.readout = SpectralReadout(bands, Q, readout=readout)
        self.gccm = GlobalColorMatrix(giv_dim, bands)
        self.lccm = LocalColorMatrixLR(dim, bands)

    def forward(self, src: torch.Tensor,
                coarse_out: torch.Tensor = None) -> torch.Tensor:
        # coarse_out is already upsampled to this level's resolution by the
        # pyramid's WaveletUpsampler (exact iDWT synthesis), so it is added
        # directly as residual guidance -- no bilinear interpolation.
        x = src if coarse_out is None else src + coarse_out

        giv = self.giv(x)                                    # [B, giv_dim]
        V = self.value_enc(x)                                # [B, bands, H, W]
        Fc = self.coeff_enc(x, giv)                          # [B, dim, H, W]
        brightness = x.mean(1, keepdim=True)                 # [B, 1, H, W]

        psi = self.core(V, Fc, brightness)                   # [B, bands, Q, H, W]
        y = self.readout(psi)                                # [B, bands, H, W]
        y = self.gccm(y, giv)                                # global cross-band
        y = self.lccm(y, Fc)                                 # local cross-band
        return y + x                                         # residual


class BCUSGSPyramid(nn.Module):
    """Banded-Corrector USGS coarse-to-fine pyramid (Tier A).

    Three coarse-to-fine levels (X/4 -> X/2 -> X) with a switchable
    recomposition, selected by ``upsample``:

      * ``wavelet``  (default): input scales are box-averaged (the wavelet LL
        grid) and each coarse spectral estimate is upsampled to the next scale
        by an exact inverse-Haar-DWT synthesis with learned high-frequency
        detail (WaveletUpsampler) -- no bilinear resampling. Inputs are
        reflect-padded to a multiple of 4 for the 2-level wavelet grid and the
        full-res output is cropped back.
      * ``bilinear`` (classic Laplacian pyramid): input scales and coarse->fine
        upsampling both use bilinear interpolation. Same Tier A core, so this
        is a clean pyramid-only A/B against the wavelet path.

    Training returns the per-scale outputs for deep supervision; inference
    returns only the full-resolution tensor.
    """

    def __init__(self, bands: int = 31,
                 depths: List[int] = [1, 2, 3],
                 Q=[4, 6, 8], M=[2, 3, 4],
                 dim: int = 64, giv_dim: int = 64,
                 use_fourier: bool = True, param_head: str = 'wavelength',
                 readout: str = 'msab', use_checkpoint: bool = True,
                 upsample: str = 'wavelet'):
        super().__init__()
        assert upsample in ('wavelet', 'bilinear')
        self.upsample = upsample
        Qs = [Q] * 3 if isinstance(Q, int) else list(Q)
        Ms = [M] * 3 if isinstance(M, int) else list(M)
        Ds = [dim] * 3 if isinstance(dim, int) else list(dim)

        def _level(idx: int) -> BCUSGSLevel:
            return BCUSGSLevel(
                bands=bands, Q=Qs[idx], M=Ms[idx], dim=Ds[idx],
                giv_dim=giv_dim, blocks=depths[idx], use_fourier=use_fourier,
                param_head=param_head, readout=readout,
                use_checkpoint=use_checkpoint)

        self.coarse = _level(0)
        self.mid = _level(1)
        self.fine = _level(2)
        if upsample == 'wavelet':
            self.up_mid = WaveletUpsampler(bands)    # X/4 -> X/2 synthesis
            self.up_fine = WaveletUpsampler(bands)   # X/2 -> X   synthesis
        else:
            self.up_mid = None
            self.up_fine = None

    def _forward_wavelet(self, src: torch.Tensor):
        H, W = src.shape[-2:]
        # Pad to a multiple of 4 so the 2-level wavelet grid round-trips exactly
        # (avg_pool /2 /4 and iDWT x2 x2 stay size-consistent for any input).
        Hp = (H + 3) // 4 * 4
        Wp = (W + 3) // 4 * 4
        if Hp != H or Wp != W:
            src = F.pad(src, (0, Wp - W, 0, Hp - H), mode='reflect')

        src2 = F.avg_pool2d(src, 2)                    # X/2 input (LL grid)
        src4 = F.avg_pool2d(src2, 2)                   # X/4 input (LL grid)

        y4 = self.coarse(src4)                         # [B, bands, H/4, W/4]
        up1 = self.up_mid(y4)                          # exact iDWT synthesis -> H/2
        y2 = self.mid(src2, coarse_out=up1)            # [B, bands, H/2, W/2]
        up0 = self.up_fine(y2)                         # exact iDWT synthesis -> H
        y = self.fine(src, coarse_out=up0)             # [B, bands, H, W]

        if Hp != H or Wp != W:
            y = y[..., :H, :W]
        return y, y2, y4

    def _forward_bilinear(self, src: torch.Tensor):
        # Classic Laplacian pyramid: bilinear analysis + bilinear synthesis.
        src2 = F.interpolate(src, scale_factor=0.5,
                             mode='bilinear', align_corners=False)
        src4 = F.interpolate(src, scale_factor=0.25,
                             mode='bilinear', align_corners=False)

        y4 = self.coarse(src4)                         # [B, bands, H/4, W/4]
        up1 = F.interpolate(y4, size=src2.shape[-2:],
                            mode='bilinear', align_corners=False)
        y2 = self.mid(src2, coarse_out=up1)            # [B, bands, H/2, W/2]
        up0 = F.interpolate(y2, size=src.shape[-2:],
                            mode='bilinear', align_corners=False)
        y = self.fine(src, coarse_out=up0)             # [B, bands, H, W]
        return y, y2, y4

    def forward(self, src: torch.Tensor):
        if self.upsample == 'wavelet':
            y, y2, y4 = self._forward_wavelet(src)
        else:
            y, y2, y4 = self._forward_bilinear(src)

        if self.training:
            return y, y2, y4          # deep supervision (fine, mid, coarse)
        return y
