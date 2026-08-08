"""KST-SAGF — a literal, hypernetwork-driven Sprecher superposition.

This is a from-the-paper rebuild of the USGS core. Where ``bc_usgs`` applied an
*independent per-band* Gaussian mixture, this model implements Sprecher's
version of Kolmogorov's superposition theorem (remonkoe.pdf, Thm 2.1)::

        f(x) = sum_{q=0}^{Qo-1} Phi_q( xi_q(x) ),
        xi_q(x) = sum_{p=1}^{n} alpha_p(x) * psi_x( x_p + a_q(x) ),

with the single inner function ``psi`` realized as a **1-D Self-Adaptive
Gaussian Field** (USGS-ru.pdf, Def. 1-2)::

        psi_x(t) = sum_{m=1}^{M} A_m(x) * exp( -(t - mu_m(x))^2 / (2 sigma_m(x)^2) ).

The four hypernetwork-driven pieces, matching the requested design:

  0. ``ChannelMixAttention``  — spectral attention that mixes the 31 source
     bands into the ``n``-D per-pixel control vector ``x`` (n = N, default 31).
  1. ``psi`` coefficients ``{A_m, mu_m, sigma_m}`` are produced *per pixel* by a
     hypernetwork (``SAGFHyperCore.psi_head``). ``psi`` itself is shared across
     all coordinates ``p`` (one inner function, Sprecher-style), but its shape
     is reconfigured at every pixel by the control features.
  2. ``alpha_p`` — the inner-sum mixing weights — come from a *separate* per-
     pixel hypernetwork (paper: fixed constants; here learned & spatial).
  3. ``a_q`` — the per-term shifts that replace the fixed ``q*a`` grid — come
     from a *separate* per-pixel hypernetwork (initialized to an even ``q*step``
     grid so it starts paper-faithful, then adapts).
  4. ``Phi_q`` — the outer functions — realized as an **MSAB-based external
     operator** (``ChiReadout``, in the spirit of MST++/hgsa_v15's ``ChiNet``):
     the ``Qo`` inner sums ``xi_q`` are concatenated with the coefficient
     features and passed through a spectral-attention (MSAB) block that expands
     ``Qo -> bands``. The pure Kolmogorov superposition ``sum_q xi_q`` is kept
     as a bare-``Parameter``-gated per-band residual so the depth-2 sum stays
     the backbone.

The coefficient (weights) path is a hgsa_v15-style encoder
(``V15CoeffEncoder``): a learned illumination sub-network gates a multi-scale
MSAB spectral transformer fused by gated ``AdvancedGFFN`` feed-forwards — the
structure that produced hgsa_v15's good results. The value path (the ``n``
Kolmogorov coordinates) is wrapped by a gated channel-mix **before**
(``val_premix``) and the inner sums ``xi`` by a gated channel-mix **after**
(``val_postmix``) the SAGF core. Global/low-rank-local color matrices, the
coarse-to-fine Laplacian/wavelet pyramid with deep supervision, and gradient
checkpointing of the ``M`` loop are preserved. A lightweight attention
(``PyramidChannelFusion``) fuses the three pyramid levels by concatenating
their channels, applying channel attention and reducing back to ``bands``.
Learnable priors are bare ``nn.Parameter`` s (survive the pipeline's generic
Conv/Linear re-init), gated by a small ``offset_scale``.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..hgsa.hgsa_hsi_v18 import (
    DegradationAwareConditioner,
    GlobalColorMatrix,
)
from .bc_usgs import (
    IlluminationEstimator,
    LocalColorMatrixLR,
    WaveletUpsampler,
    _make_msab,
)


class ChannelMixAttention(nn.Module):
    """Step 0 — mix the source bands into the ``n``-D control vector ``x``.

    Spectral (MSAB) attention over the input bands produces the per-pixel
    Kolmogorov coordinate vector ``x``. The output is squashed with a sigmoid
    into the paper's unit-cube domain ``[0, 1]^n`` (Thm 2.1 is stated on
    ``I^n``), which is exactly the range the SAGF ``mu`` grid (init ``[0.1,
    0.9]``) and the shift priors expect. Bounding the coordinate is essential:
    it keeps the Gaussian argument ``(x_p + a_q - mu) / sigma`` well-conditioned
    regardless of the pipeline's generic Conv/Linear re-init and of the
    coarse-to-fine residual accumulation across pyramid levels.
    """

    def __init__(self, bands: int, n: int, dim: int = 64,
                 num_blocks: int = 2, heads: int = 4):
        super().__init__()
        self.bands, self.n = bands, n
        self.stem = nn.Conv2d(bands, dim, 3, padding=1)
        self.body = _make_msab(dim, num_blocks=num_blocks, heads=heads)
        self.head = nn.Conv2d(dim, n, 3, padding=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        f = self.body(self.stem(img))
        return torch.sigmoid(self.head(f))                   # x in (0, 1)^n


class SAGFHyperCore(nn.Module):
    """Steps 1-3 — the Sprecher inner sums driven by three hypernetworks.

    From the coefficient/illumination features it predicts, *per pixel*:
      * ``psi_head`` -> the 1-D SAGF coefficients ``{A_m, mu_m, sigma_m}`` that
        define the (shared) inner function ``psi_x`` at this pixel  (step 1);
      * ``alpha_head`` -> the ``n`` inner-sum weights ``alpha_p``      (step 2);
      * ``shift_head`` -> the ``Qo`` per-term shifts ``a_q``           (step 3).

    It then evaluates, for every outer term ``q``::

        xi_q = sum_p alpha_p * psi_x( x_p + a_q ),
        psi_x(t) = sum_m A_m * exp( -(t - mu_m)^2 / (2 sigma_m^2) ),

    returning the ``Qo`` inner sums ``xi`` [B, Qo, H, W]. The ``m`` loop is
    gradient-checkpointed: only one Gaussian's ``[B, n, Qo, H, W]`` argument
    tensor is live at a time.
    """

    def __init__(self, n: int, Qo: int, M: int, feat_dim: int,
                 illu_dim: int, shift_step: float = 0.03,
                 use_checkpoint: bool = True):
        super().__init__()
        self.n, self.Qo, self.M = n, Qo, M
        self.use_checkpoint = use_checkpoint
        in_dim = feat_dim + illu_dim

        # --- Hypernetwork heads (per-pixel 1x1 conv on the control features) ---
        # psi (SAGF) coefficients: 3*M maps (A, mu, sigma) per pixel.
        self.psi_head = nn.Conv2d(in_dim, 3 * M, 1)
        # alpha_p mixing weights.
        self.alpha_head = nn.Conv2d(in_dim, n, 1)
        # a_q shifts.
        self.shift_head = nn.Conv2d(in_dim, Qo, 1)

        # --- Learnable priors (bare Parameters: survive pipeline re-init) ---
        # Evenly-spaced mu grid over [0.1, 0.9] (SAGF centers span the coord range).
        mu_grid = torch.linspace(0.1, 0.9, M).view(1, M, 1, 1)
        self.mu_base = nn.Parameter(mu_grid.clone())
        self.sigma_base = nn.Parameter(torch.full((1, M, 1, 1), 0.15))
        self.amp_base = nn.Parameter(0.1 * torch.randn(1, M, 1, 1))
        # alpha prior: uniform 1 (paper alpha_1 = 1, decaying; start uniform).
        self.alpha_base = nn.Parameter(torch.ones(1, n, 1, 1))
        # shift prior: even q*step grid (paper q*a), q = 0..Qo-1.
        shift_grid = (torch.arange(Qo).float() * shift_step).view(1, Qo, 1, 1)
        self.shift_base = nn.Parameter(shift_grid.clone())
        # Global tameness gate on all per-pixel offsets (robust to re-init).
        self.offset_scale = nn.Parameter(torch.tensor(0.1))

    def _params(self, ctrl: torch.Tensor):
        """Predict the per-pixel SAGF / alpha / shift fields from features."""
        s = self.offset_scale
        p = self.psi_head(ctrl)                               # [B, 3M, H, W]
        B, _, H, W = p.shape
        p = p.view(B, 3, self.M, H, W) * s
        amp = self.amp_base + p[:, 0]                         # [B, M, H, W]
        mu = self.mu_base + p[:, 1]
        # sigma floored well above 0 so (x - mu)/sigma cannot overflow fp16.
        sigma = F.softplus(self.sigma_base + p[:, 2]) + 5e-2
        alpha = self.alpha_base + s * self.alpha_head(ctrl)   # [B, n, H, W]
        shift = self.shift_base + s * self.shift_head(ctrl)   # [B, Qo, H, W]
        return amp, mu, sigma, alpha, shift

    def _gauss_term(self, m: int, xs: torch.Tensor, alpha: torch.Tensor,
                    amp: torch.Tensor, mu: torch.Tensor,
                    sigma: torch.Tensor) -> torch.Tensor:
        """One SAGF expert's contribution to every xi_q (summed over p)."""
        a = amp[:, m].unsqueeze(1).unsqueeze(1)               # [B,1,1,H,W]
        mu_m = mu[:, m].unsqueeze(1).unsqueeze(1)
        sig_m = sigma[:, m].unsqueeze(1).unsqueeze(1)
        # Clamp the normalized argument to +-8 sigma (Gaussian is ~1e-14 there)
        # so ((xs-mu)/sigma)^2 never overflows fp16 and the backward pass
        # cannot produce inf/0*inf -> NaN.
        z = ((xs - mu_m) / sig_m).clamp(-8.0, 8.0)
        g = a * torch.exp(-0.5 * z * z)                       # [B,n,Qo,H,W]
        return (alpha.unsqueeze(2) * g).sum(dim=1)            # [B, Qo, H, W]

    def forward(self, x: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        B, n, H, W = x.shape
        amp, mu, sigma, alpha, shift = self._params(ctrl)
        # Shifted arguments xs[b,p,q,h,w] = x_p + a_q   (the "x_p + q a" grid).
        xs = x.unsqueeze(2) + shift.unsqueeze(1)             # [B, n, Qo, H, W]

        xi = x.new_zeros(B, self.Qo, H, W)
        for m in range(self.M):
            if self.use_checkpoint and self.training:
                contrib = checkpoint(self._gauss_term, m, xs, alpha, amp, mu,
                                     sigma, use_reentrant=False)
            else:
                contrib = self._gauss_term(m, xs, alpha, amp, mu, sigma)
            xi = xi + contrib
        return xi                                            # [B, Qo, H, W]


class AdvancedGFFN(nn.Module):
    """Gated feed-forward (hgsa_v15 ``Advanced_GFFN``), generalized to HSI.

    ``project_in`` doubles the width; a squeeze-excite spectral calibration
    re-weights the channels; the two halves are combined multiplicatively as a
    depthwise-3x3 branch gated by ``sigmoid`` of a depthwise-5x5 branch. This is
    the proven feed-forward that fused hgsa_v15's multi-scale encoder features.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.project_in = nn.Conv2d(in_dim, out_dim * 2, 1)
        self.cal = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim * 2, out_dim * 2, 1), nn.Sigmoid())
        self.dw3 = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim)
        self.dw5 = nn.Conv2d(out_dim, out_dim, 5, padding=2, groups=out_dim)
        self.project_out = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.project_in(x)
        c = c * self.cal(c)
        x1, x2 = c.chunk(2, dim=1)
        return self.project_out(self.dw3(x1) * torch.sigmoid(self.dw5(x2)))


class GatedChannelMix(nn.Module):
    """Lightweight, gated MSAB channel mixer (near-identity at init).

    Used to wrap the **value path** (the ``n`` Kolmogorov coordinates before the
    SAGF core, and the ``Qo`` inner sums ``xi`` after it). MSAB is internally
    residual, so ``x + gamma * (MSAB(x) - x)`` isolates its delta and the bare
    ``Parameter`` gate keeps the mixer reducible to a no-op / stable at init.
    """

    def __init__(self, ch: int, heads: int = 4, num_blocks: int = 1):
        super().__init__()
        self.mix = _make_msab(ch, num_blocks=num_blocks, heads=heads)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * (self.mix(x) - x)


class ChiReadout(nn.Module):
    """Step 4 — the MSAB-based external outer operator ``Phi_q`` (χ / ChiNet).

    In the spirit of MST++ and hgsa_v15's ``ChiNet``, the outer stage is a
    spectral-attention (MSAB) network rather than a bank of scalar 1-D
    functions. The ``Qo`` inner sums ``xi_q`` are concatenated with the
    coefficient features, projected to a hidden width, mixed by an MSAB block,
    then (with an ``xi`` skip) projected to the ``bands`` output channels::

        h = MSAB( conv1x1( [xi ; feat] ) )
        y = conv1x1( [h ; xi] )

    The cross-channel coupling that expands ``Qo -> bands`` lives entirely in
    this external operator, matching the requested design.
    """

    def __init__(self, bands: int, Qo: int, feat_dim: int,
                 hidden: int = 64, num_blocks: int = 2, heads: int = 4):
        super().__init__()
        self.pre = nn.Conv2d(Qo + feat_dim, hidden, 1)
        self.msab = _make_msab(hidden, num_blocks=num_blocks, heads=heads)
        self.post = nn.Conv2d(hidden + Qo, bands, 1)

    def forward(self, xi: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        h = self.pre(torch.cat([xi, feat], dim=1))
        h = self.msab(h)
        return self.post(torch.cat([h, xi], dim=1))


class V15CoeffEncoder(nn.Module):
    """Coefficient (weights) path — a hgsa_v15 ``Encoder2D``-style encoder.

    Mirrors the structure that produced hgsa_v15's good results, generalized
    from RGB to ``bands`` inputs with the *tested* MSAB spectral-attention
    block:

      * a learned illumination sub-network (Retinex-style) whose features gate
        the stem, giving illumination-aware conditioning;
      * a multi-scale spectral transformer — full-, half- and quarter-scale
        MSAB branches (down/up by average pooling + bilinear) — for local and
        global spatial context;
      * gated ``AdvancedGFFN`` feed-forwards to fuse the scales.

    Returns ``(feat[B, dim, H, W], illu_fea[B, illu_dim, H, W])`` — the two
    signals the SAGF hypernetwork heads read from.
    """

    def __init__(self, bands: int, dim: int, illu_dim: int = 32,
                 blocks: int = 2, heads: int = 4):
        super().__init__()
        self.illu = IlluminationEstimator(bands, illu_dim)
        self.stem = nn.Conv2d(bands, dim, 3, padding=1)
        self.illu_gate = nn.Conv2d(illu_dim, dim, 1)
        self.enc0 = _make_msab(dim, num_blocks=blocks, heads=heads)
        self.enc1 = _make_msab(dim, num_blocks=blocks, heads=heads)
        self.enc2 = _make_msab(dim, num_blocks=blocks, heads=heads)
        self.fuse = AdvancedGFFN(dim * 3, dim)
        self.norm = nn.GroupNorm(min(4, dim), dim)
        self.gffn = AdvancedGFFN(dim, dim)

    def forward(self, x: torch.Tensor):
        illu_fea, _ = self.illu(x)                           # [B, illu_dim, H, W]
        f = self.stem(x) * torch.sigmoid(self.illu_gate(illu_fea))
        f0 = self.enc0(f)
        f1 = self.enc1(F.avg_pool2d(f0, 2))
        f1 = F.interpolate(f1, size=f0.shape[-2:],
                           mode='bilinear', align_corners=False)
        f2 = self.enc2(F.avg_pool2d(f0, 4))
        f2 = F.interpolate(f2, size=f0.shape[-2:],
                           mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([f0, f1, f2], dim=1))
        return self.gffn(self.norm(fused)) + fused, illu_fea


class PyramidChannelFusion(nn.Module):
    """Request 5 — lightweight attention fusion of the Laplacian pyramid.

    The three pyramid levels' outputs (all brought to the finest resolution)
    are fused by *concatenating their channels*, applying a squeeze-excite
    channel attention over the concatenation, and reducing the ``n_levels *
    bands`` channels back to ``bands`` with a 1x1 convolution. Added to the
    finest output as a bare-``Parameter``-gated residual (reducible to a no-op).
    """

    def __init__(self, bands: int, n_levels: int = 3):
        super().__init__()
        c = bands * n_levels
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(1, c // 4), 1), nn.GELU(),
            nn.Conv2d(max(1, c // 4), c, 1), nn.Sigmoid())
        self.reduce = nn.Conv2d(c, bands, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(feats, dim=1)                          # [B, n*bands, H, W]
        x = x * self.attn(x)
        return feats[0] + self.gamma * self.reduce(x)


class KSTSagfLevel(nn.Module):
    """One coarse-to-fine pyramid level of the KST-SAGF operator."""

    def __init__(self, bands: int = 31, n: int = 31, Qo: int = 8, M: int = 4,
                 dim: int = 96, giv_dim: int = 64, blocks: int = 2,
                 chi_hidden: int = 64, chi_heads: int = 4,
                 use_checkpoint: bool = True, illu_dim: int = 32):
        super().__init__()
        self.bands = bands
        self.giv = DegradationAwareConditioner(bands, giv_dim)
        # --- Value path: img -> n Kolmogorov coordinates, wrapped by gated
        # channel-mixers before (val_premix) and after (val_postmix, on xi) the
        # SAGF core. These mix ONLY the value path (not the coefficient path).
        self.channel_mix = ChannelMixAttention(bands, n, dim=dim)
        self.val_premix = GatedChannelMix(n, heads=1)
        self.val_postmix = GatedChannelMix(Qo, heads=chi_heads)
        # --- Coefficient path: hgsa_v15-style illumination-gated multi-scale
        # MSAB encoder (the structure that produced hgsa_v15's good results).
        self.coeff_enc = V15CoeffEncoder(
            bands, dim, illu_dim=illu_dim, blocks=max(1, blocks))
        # Steps 1-3: Sprecher inner sums with the shared SAGF inner function.
        self.core = SAGFHyperCore(
            n, Qo, M, feat_dim=dim, illu_dim=illu_dim,
            use_checkpoint=use_checkpoint)
        # Step 4: MSAB-based external outer operator Phi_q (Qo -> bands) ...
        self.chi = ChiReadout(
            bands, Qo, feat_dim=dim, hidden=chi_hidden, heads=chi_heads)
        # ... plus the pure Kolmogorov superposition sum_q xi_q as a gated
        # per-band residual, so the depth-2 sum stays the backbone.
        self.kst_gamma = nn.Parameter(torch.tensor(0.1))
        # Cross-band polish (residual around identity), as in bc_usgs.
        self.gccm = GlobalColorMatrix(giv_dim, bands)
        self.lccm = LocalColorMatrixLR(dim, bands)
        # Bounded, bare-Parameter-gated correction (near-identity at init;
        # tanh soft-clamps so the level can never blow up under the pipeline's
        # generic Conv/Linear re-init).
        self.out_gate = nn.Parameter(torch.tensor(0.2))

    def forward(self, src: torch.Tensor,
                coarse_out: torch.Tensor = None) -> torch.Tensor:
        x_in = src if coarse_out is None else src + coarse_out

        giv = self.giv(x_in)                                 # [B, giv_dim]
        coords = self.channel_mix(x_in)                      # x [B, n, H, W]
        coords = self.val_premix(coords)                     # value-path pre-mix
        feat, illu_fea = self.coeff_enc(x_in)                # coefficient path
        ctrl = torch.cat([feat, illu_fea], dim=1)            # hypernet input

        xi = self.core(coords, ctrl)                         # [B, Qo, H, W]
        xi = self.val_postmix(xi)                            # value-path post-mix
        y = self.chi(xi, feat)                               # MSAB Phi_q -> bands
        y = y + self.kst_gamma * xi.sum(1, keepdim=True)     # pure-KST backbone
        y = self.gccm(y, giv)                                # global cross-band
        y = self.lccm(y, feat)                               # local cross-band
        y = self.out_gate * torch.tanh(y)                    # bounded correction
        return y + x_in                                      # residual


class KSTSagfPyramid(nn.Module):
    """KST-SAGF coarse-to-fine (Laplacian / wavelet) pyramid.

    Three levels (X/4 -> X/2 -> X); each level's coarse estimate guides the
    next via ``bilinear`` interpolation (classic Laplacian pyramid) or exact
    inverse-Haar-DWT ``wavelet`` synthesis. Training returns the per-scale
    outputs for deep supervision; inference returns the full-resolution tensor.
    """

    def __init__(self, bands: int = 31, n: int = 31,
                 depths: List[int] = [1, 2, 3],
                 Qo=[12, 16, 24], M=[3, 4, 6],
                 dim: int = 96, giv_dim: int = 64,
                 chi_hidden: int = 64, chi_heads: int = 4,
                 use_checkpoint: bool = True,
                 upsample: str = 'bilinear', illu_dim: int = 32,
                 pyramid_fusion: bool = True):
        super().__init__()
        assert upsample in ('wavelet', 'bilinear')
        self.upsample = upsample
        Qs = [Qo] * 3 if isinstance(Qo, int) else list(Qo)
        Ms = [M] * 3 if isinstance(M, int) else list(M)
        Ds = [dim] * 3 if isinstance(dim, int) else list(dim)

        def _level(idx: int) -> KSTSagfLevel:
            return KSTSagfLevel(
                bands=bands, n=n, Qo=Qs[idx], M=Ms[idx], dim=Ds[idx],
                giv_dim=giv_dim, blocks=depths[idx], chi_hidden=chi_hidden,
                chi_heads=chi_heads, use_checkpoint=use_checkpoint,
                illu_dim=illu_dim)

        self.coarse = _level(0)
        self.mid = _level(1)
        self.fine = _level(2)
        # Request 5: lightweight attention fusion of the three pyramid levels.
        self.fusion = PyramidChannelFusion(bands) if pyramid_fusion else None
        if upsample == 'wavelet':
            self.up_mid = WaveletUpsampler(bands)
            self.up_fine = WaveletUpsampler(bands)
        else:
            self.up_mid = None
            self.up_fine = None

    def _forward_wavelet(self, src: torch.Tensor):
        H, W = src.shape[-2:]
        Hp = (H + 3) // 4 * 4
        Wp = (W + 3) // 4 * 4
        if Hp != H or Wp != W:
            src = F.pad(src, (0, Wp - W, 0, Hp - H), mode='reflect')
        src2 = F.avg_pool2d(src, 2)
        src4 = F.avg_pool2d(src2, 2)
        y4 = self.coarse(src4)
        y2 = self.mid(src2, coarse_out=self.up_mid(y4))
        y = self.fine(src, coarse_out=self.up_fine(y2))
        if self.fusion is not None:
            y2u = F.interpolate(y2, size=y.shape[-2:],
                                mode='bilinear', align_corners=False)
            y4u = F.interpolate(y4, size=y.shape[-2:],
                                mode='bilinear', align_corners=False)
            y = self.fusion([y, y2u, y4u])                   # pyramid fusion
        if Hp != H or Wp != W:
            y = y[..., :H, :W]
        return y, y2, y4

    def _forward_bilinear(self, src: torch.Tensor):
        src2 = F.interpolate(src, scale_factor=0.5,
                             mode='bilinear', align_corners=False)
        src4 = F.interpolate(src, scale_factor=0.25,
                             mode='bilinear', align_corners=False)
        y4 = self.coarse(src4)
        up1 = F.interpolate(y4, size=src2.shape[-2:],
                            mode='bilinear', align_corners=False)
        y2 = self.mid(src2, coarse_out=up1)
        up0 = F.interpolate(y2, size=src.shape[-2:],
                            mode='bilinear', align_corners=False)
        y = self.fine(src, coarse_out=up0)
        if self.fusion is not None:
            y2u = F.interpolate(y2, size=y.shape[-2:],
                                mode='bilinear', align_corners=False)
            y4u = F.interpolate(y4, size=y.shape[-2:],
                                mode='bilinear', align_corners=False)
            y = self.fusion([y, y2u, y4u])                   # pyramid fusion
        return y, y2, y4

    def forward(self, src: torch.Tensor):
        if self.upsample == 'wavelet':
            y, y2, y4 = self._forward_wavelet(src)
        else:
            y, y2, y4 = self._forward_bilinear(src)
        if self.training:
            return y, y2, y4
        return y
