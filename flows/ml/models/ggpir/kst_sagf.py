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
  4. ``Phi_q`` — the outer functions — with two switchable forms (``readout``):
     * ``kan`` (default, ``PerBandKANReadout``): the paper-literal vector form,
       a full ``Qo x bands`` grid of independent 1-D functions ``Phi_{q,o}``
       (a Kolmogorov-Arnold layer), ``y_o = sum_q Phi_{q,o}(xi_q)``. No
       cross-channel attention bottleneck — highest capacity.
     * ``mixer``: a single shared 1-D ``Phi_q`` (``Phi1D``) applied per term,
       followed by a *separate* external multi-head cross-channel attention
       (``ExternalChannelMixer``) that expands ``Qo -> bands``, plus the pure
       Kolmogorov sum ``sum_q Phi_q(xi_q)`` as a per-band residual.

Everything else that made the previous design train — the coefficient encoder
(MSAB + multi-scale dilation + global Fourier + FiLM(GIV) + illumination),
global/low-rank-local color matrices, the coarse-to-fine Laplacian/​wavelet
pyramid with deep supervision, and gradient checkpointing of the ``M`` loop —
is preserved. Learnable priors are bare ``nn.Parameter`` s (they survive the
pipeline's generic Conv/Linear re-init), gated by a small ``offset_scale``.

Capacity levers (all bare-``Parameter``-gated residuals, so each is reducible
to a no-op and the Sprecher superposition stays the backbone):

  A. ``PostKSTRefine`` — post-superposition MSAB cross-band re-coupling (+ an
     optional depthwise-3x3 spatial mixer) on the reconstructed spectra.
  B. ``SAGFHyperCore(head_spatial=True)`` — a gated depthwise-3x3 mixer on the
     control features before the pointwise coefficient heads (enriches only the
     hypernetwork; the ``xi_q`` evaluation is unchanged).
  C. ``PerBandKANReadout(spectral_prior=True)`` — a smooth per-band baseline
     from a learned band embedding plus gated band-axis smoothing.
  D. larger ``M`` — a richer single shared inner function ``psi``.
  E. ``FineRefiner`` — a full-res residual conv block on the finest output.

A's spatial half and E introduce a receptive field, so they are auxiliary
*post-KST* terms (outside Thm 2.1's pointwise property); B, C, D and A's
cross-band half stay within the superposition's form.
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
    BCCoeffEncoder,
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
                 giv_dim: int, illu_dim: int, shift_step: float = 0.03,
                 use_checkpoint: bool = True, head_spatial: bool = True):
        super().__init__()
        self.n, self.Qo, self.M = n, Qo, M
        self.use_checkpoint = use_checkpoint
        in_dim = feat_dim + illu_dim

        # (B) Spatially-aware coefficient head: a gated depthwise-3x3 + 1x1
        # mixer on the control features before the pointwise parameter heads,
        # so psi/alpha/shift adapt to *local* structure (restores the
        # ExpertParamHead spatial capacity that the plain 1x1 heads dropped).
        # This enriches only the hypernetwork that *predicts* the coefficients;
        # the Sprecher evaluation xi_q = sum_p alpha_p psi(x_p + a_q) below is
        # unchanged, so Theorem 2.1's form is preserved. Gated by a bare
        # Parameter (survives re-init; reducible to the pure 1x1 head).
        if head_spatial:
            self.head_mix = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim),
                nn.GroupNorm(1, in_dim), nn.GELU(),
                nn.Conv2d(in_dim, in_dim, 1))
            self.head_gamma = nn.Parameter(torch.tensor(0.1))
        else:
            self.head_mix = None
            self.head_gamma = None

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
        # (B) gated spatial enrichment of the control features.
        if self.head_mix is not None:
            ctrl = ctrl + self.head_gamma * self.head_mix(ctrl)
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


class Phi1D(nn.Module):
    """Step 4a — the paper's 1-D outer functions ``Phi_q : R -> R``.

    Exactly as in remonkoe.pdf Thm 2.1, each superposition term ``q`` gets its
    own *scalar* outer function applied elementwise to the inner sum ``xi_q``
    (no cross-channel coupling here — that is deferred to the external mixer).
    Each ``Phi_q`` is a tiny per-``q`` 1-D MLP (``1 -> h -> 1`` with GELU),
    written as a residual around the identity::

        u_q = xi_q + gamma * ( W2_q . GELU(W1_q * xi_q + b1_q) + b2_q )

    The residual/identity start (small ``gamma``) keeps the model close to a
    plain linear outer function at init — stable — while giving each term a
    genuine, independent 1-D nonlinearity that grows during training. The
    per-``q`` parameters are bare ``Parameter`` s (survive the pipeline's
    generic re-init). ``sum_q u_q`` is the pure Kolmogorov superposition scalar.
    """

    def __init__(self, Qo: int, hidden: int = 16):
        super().__init__()
        self.Qo, self.h = Qo, hidden
        self.w1 = nn.Parameter(torch.randn(Qo, hidden))
        self.b1 = nn.Parameter(torch.zeros(Qo, hidden))
        self.w2 = nn.Parameter(torch.randn(Qo, hidden) / hidden ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(Qo))
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        B, Qo, H, W = xi.shape
        t = xi.unsqueeze(-1)                                  # [B, Qo, H, W, 1]
        w1 = self.w1.view(1, Qo, 1, 1, self.h)
        b1 = self.b1.view(1, Qo, 1, 1, self.h)
        hdn = F.gelu(t * w1 + b1)                             # [B, Qo, H, W, h]
        w2 = self.w2.view(1, Qo, 1, 1, self.h)
        phi = (hdn * w2).sum(-1) + self.b2.view(1, Qo, 1, 1)  # [B, Qo, H, W]
        return xi + self.gamma * phi                         # u_q


class ExternalChannelMixer(nn.Module):
    """Step 4b — the *extra*, external cross-channel function (attention-based).

    This is the operator that turns the ``Qo`` post-outer values
    ``u_q = Phi_q(xi_q)`` into the ``bands`` output channels. It is deliberately
    kept *separate* from ``Phi_q`` (which is 1-D and per-``q``): all cross-
    channel coupling lives here. ``bands`` learned band-queries (offset by a
    global-scene / GIV-conditioned term) cross-attend, with ``heads`` heads,
    over the ``Qo`` tokens::

        key_q, val_q = Embed(u_q) + posemb_q                 (linear embedding)
        A[o, q]      = softmax_q( <query_o, key_q> / sqrt(d_head) )
        y_o          = out_proj( sum_q A[o, q] * val_q ) + bias_o(giv)

    The token embedding is purely linear (the nonlinearity now lives in
    ``Phi_q``); the per-``q`` positional embedding lets a given band prefer
    specific superposition terms. Everything is per-pixel (H, W folded into the
    token batch).
    """

    def __init__(self, Qo: int, bands: int, giv_dim: int, d: int = 32,
                 heads: int = 4):
        super().__init__()
        heads = max(1, heads)
        while d % heads != 0:
            heads -= 1
        self.Qo, self.bands, self.d = Qo, bands, d
        self.heads, self.dh = heads, d // heads
        self.scale = self.dh ** -0.5
        # Linear token embedding of the scalar u_q -> key & value.
        self.embed = nn.Linear(1, 2 * d)
        self.q_pos = nn.Parameter(0.02 * torch.randn(Qo, 2 * d))
        # Band queries: learned base + global-scene (GIV) conditioning.
        self.band_base = nn.Parameter(0.02 * torch.randn(bands, d))
        self.q_from_giv = nn.Linear(giv_dim, bands * d)
        nn.init.zeros_(self.q_from_giv.weight)
        nn.init.zeros_(self.q_from_giv.bias)
        # Value -> per-band scalar, plus a global-scene bias per band.
        self.out_proj = nn.Linear(d, 1)
        self.band_bias = nn.Linear(giv_dim, bands)
        nn.init.zeros_(self.band_bias.weight)
        nn.init.zeros_(self.band_bias.bias)

    def forward(self, u: torch.Tensor, giv: torch.Tensor) -> torch.Tensor:
        B, Qo, H, W = u.shape
        d, hh, e = self.d, self.heads, self.dh
        t = u.permute(0, 2, 3, 1).reshape(B, H * W, Qo, 1)   # scalar tokens
        kv = self.embed(t) + self.q_pos                      # [B, HW, Qo, 2d]
        k, v = kv[..., :d], kv[..., d:]
        q = self.band_base.unsqueeze(0) + \
            self.q_from_giv(giv).view(B, self.bands, d)      # [B, bands, d]
        # Split into heads: e = d // heads.
        N = H * W
        k = k.view(B, N, Qo, hh, e)
        v = v.view(B, N, Qo, hh, e)
        qh = q.view(B, self.bands, hh, e)
        scores = torch.einsum('bnqhe,bohe->bnhoq', k, qh) * self.scale
        attn = torch.softmax(scores, dim=-1)                 # over Qo
        out = torch.einsum('bnhoq,bnqhe->bnohe', attn, v)    # [B,HW,bands,hh,e]
        out = out.reshape(B, N, self.bands, d)
        y = self.out_proj(out).squeeze(-1)                   # [B, HW, bands]
        y = y + self.band_bias(giv).unsqueeze(1)
        return y.reshape(B, H, W, self.bands).permute(0, 3, 1, 2)


class PerBandKANReadout(nn.Module):
    """Step 4 (KAN grid) — per-band, per-term 1-D outer functions ``Phi_{q,o}``.

    The most paper-literal outer stage for a vector output: instead of a single
    shared ``Phi_q`` followed by a cross-channel mixer, this is a full
    ``Qo x bands`` grid of independent 1-D functions (a Kolmogorov-Arnold
    Network layer)::

        y_o = sum_{q=0}^{Qo-1} Phi_{q,o}( xi_q ),

    with each edge a KAN spline-plus-base::

        Phi_{q,o}(xi) = wb_{q,o} * SiLU(xn) + sum_{j=1}^{G} c_{q,o,j} * rbf_j(xn),
        xn = tanh(in_scale * xi)  in (-1, 1),

    where ``rbf_j`` are ``G`` Gaussian bases on a fixed ``[-1, 1]`` grid. The
    ``tanh`` squashing fixes the spline domain regardless of the (unbounded)
    ``xi`` range, and every learnable weight is a bare ``Parameter`` (survives
    the pipeline's generic re-init). Removing the ``Qo -> bands`` attention
    bottleneck lets each band read the superposition terms with its own set of
    nonlinearities — higher capacity, the direction toward PSNR 42.
    """

    def __init__(self, Qo: int, bands: int, num_basis: int = 8,
                 spectral_prior: bool = True, emb_dim: int = 16):
        super().__init__()
        self.Qo, self.bands, self.G = Qo, bands, num_basis
        centers = torch.linspace(-1.0, 1.0, num_basis)
        self.register_buffer('centers', centers.view(1, 1, num_basis, 1, 1))
        self.width = 2.0 / max(1, num_basis - 1)
        self.in_scale = nn.Parameter(torch.tensor(1.0))
        # Per-edge KAN coefficients (bare Parameters, survive re-init).
        self.coef = nn.Parameter(
            (0.1 / num_basis ** 0.5) * torch.randn(Qo, bands, num_basis))
        self.w_base = nn.Parameter(0.1 * torch.randn(Qo, bands))
        self.bias = nn.Parameter(torch.zeros(bands))
        # (C) Spectral-smoothness prior. HSI bands are highly correlated, so we
        # add (i) a smooth per-band baseline from a learned low-dim band
        # embedding (adjacent bands share structure through the shared MLP) and
        # (ii) a gated fixed 3-tap smoothing along the band axis. Both are
        # pointwise cross-band operations (they stay within the "sum of
        # univariate functions of xi_q" spirit of Thm 2.1) and the smoothing is
        # a bare-Parameter-gated residual (reducible to no-op).
        if spectral_prior:
            self.band_emb = nn.Parameter(0.02 * torch.randn(bands, emb_dim))
            self.band_mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2), nn.GELU(),
                nn.Linear(emb_dim * 2, 1))
            self.smooth_gamma = nn.Parameter(torch.tensor(0.05))
        else:
            self.band_emb = None
            self.band_mlp = None
            self.smooth_gamma = None

    @staticmethod
    def _band_smooth(y: torch.Tensor) -> torch.Tensor:
        # Replicate-padded 3-tap [0.25, 0.5, 0.25] smoothing over the band axis.
        yl = torch.cat([y[:, :1], y[:, :-1]], dim=1)
        yr = torch.cat([y[:, 1:], y[:, -1:]], dim=1)
        return 0.25 * yl + 0.5 * y + 0.25 * yr

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        B, Qo, H, W = xi.shape
        xn = torch.tanh(self.in_scale * xi)                  # [B, Qo, H, W]
        rbf = torch.exp(
            -((xn.unsqueeze(2) - self.centers) / self.width) ** 2)  # [B,Qo,G,H,W]
        y = torch.einsum('qoj,bqjhw->bohw', self.coef, rbf)
        y = y + torch.einsum('qo,bqhw->bohw', self.w_base, F.silu(xn))
        y = y + self.bias.view(1, self.bands, 1, 1)          # [B, bands, H, W]
        if self.band_mlp is not None:                        # (C) spectral prior
            base = self.band_mlp(self.band_emb).view(1, self.bands, 1, 1)
            y = y + base
            y = y + self.smooth_gamma * (self._band_smooth(y) - y)
        return y


class PostKSTRefine(nn.Module):
    """(A) Post-superposition refinement — cross-band + spatial re-coupling.

    The KST outer stage is pointwise in ``xi``; it cannot re-couple the output
    bands spatially or restore cross-band correlation beyond what the shared
    ``xi_q`` carry. This module adds, as **bare-Parameter-gated residuals** on
    the reconstructed spectra ``y`` (so it is reducible to a no-op and the model
    still starts essentially as the pure superposition):

      * ``cb`` — an MSAB cross-band mixer (restores band correlation the
        per-band outer functions miss). This is the plateau-breaker.
      * ``sp`` — a depthwise-3x3 + 1x1 spatial mixer. NOTE: this introduces a
        receptive field, so it is an *auxiliary, post-KST* term that steps
        outside Theorem 2.1's pointwise property; the ``g_sp`` gate keeps it a
        small, disableable correction on top of the superposition backbone.
    """

    def __init__(self, bands: int, spatial: bool = True,
                 crossband: bool = True):
        super().__init__()
        self.cb = _make_msab(bands, num_blocks=1, heads=1) if crossband else None
        self.g_cb = nn.Parameter(torch.tensor(0.1)) if crossband else None
        if spatial:
            self.sp = nn.Sequential(
                nn.Conv2d(bands, bands, 3, padding=1, groups=bands),
                nn.GroupNorm(1, bands), nn.GELU(),
                nn.Conv2d(bands, bands, 1))
            self.g_sp = nn.Parameter(torch.tensor(0.05))
        else:
            self.sp = None
            self.g_sp = None

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.cb is not None:
            # MSAB is internally residual; (cb(y) - y) isolates its delta so the
            # gate g_cb cleanly scales (and can zero) the cross-band correction.
            y = y + self.g_cb * (self.cb(y) - y)
        if self.sp is not None:
            y = y + self.g_sp * self.sp(y)
        return y


class FineRefiner(nn.Module):
    """(E) Full-resolution residual refiner for the finest pyramid output.

    A small 3-conv block that injects high-frequency spatial detail into the
    final full-res reconstruction, as a bare-Parameter-gated residual. Like the
    spatial half of ``PostKSTRefine`` this is an auxiliary, post-KST spatial
    term (outside the pointwise Thm 2.1 form) but is reducible to a no-op via
    its gate, so the superposition remains the backbone.
    """

    def __init__(self, bands: int, hidden: int = None):
        super().__init__()
        hidden = hidden or bands * 2
        self.body = nn.Sequential(
            nn.Conv2d(bands, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, bands, 3, padding=1))
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y + self.gamma * self.body(y)


class KSTSagfLevel(nn.Module):
    """One coarse-to-fine pyramid level of the KST-SAGF operator."""

    def __init__(self, bands: int = 31, n: int = 31, Qo: int = 8, M: int = 4,
                 dim: int = 96, giv_dim: int = 64, blocks: int = 2,
                 phi_dim: int = 64, phi_heads: int = 4, readout: str = 'kan',
                 kan_basis: int = 8, use_fourier: bool = True,
                 use_checkpoint: bool = True, illu_dim: int = 32,
                 head_spatial: bool = True, spectral_prior: bool = True,
                 post_refine: bool = True, post_spatial: bool = True):
        super().__init__()
        assert readout in ('kan', 'mixer')
        self.bands = bands
        self.readout = readout
        self.giv = DegradationAwareConditioner(bands, giv_dim)
        # Step 0: control-vector attention (img -> x, n coordinates).
        self.channel_mix = ChannelMixAttention(bands, n, dim=dim)
        # Hypernetwork context (MSAB + dilation + Fourier + FiLM + illumination).
        self.coeff_enc = BCCoeffEncoder(
            bands, dim, giv_dim, illu_dim=illu_dim, num_blocks=max(1, blocks),
            use_fourier=use_fourier)
        # Steps 1-3: Sprecher inner sums with SAGF psi. (B) spatial hyper-heads.
        self.core = SAGFHyperCore(
            n, Qo, M, feat_dim=dim, giv_dim=giv_dim, illu_dim=illu_dim,
            use_checkpoint=use_checkpoint, head_spatial=head_spatial)
        # Step 4 — outer stage. Two switchable forms:
        if readout == 'kan':
            # 'kan': a full Qo x bands grid of 1-D functions Phi_{q,o}
            # (paper-literal for a vector output; no attention bottleneck).
            # (C) spectral-smoothness prior baked into the read-out.
            self.kan = PerBandKANReadout(
                Qo, bands, num_basis=kan_basis, spectral_prior=spectral_prior)
        else:
            # 'mixer': shared 1-D Phi_q (per term) + external multi-head
            # cross-channel attention that expands Qo -> bands.
            self.phi = Phi1D(Qo)
            self.mixer = ExternalChannelMixer(
                Qo, bands, giv_dim, d=phi_dim, heads=phi_heads)
            # Direct Kolmogorov superposition term sum_q Phi_q(xi_q), broadcast
            # to every band as a residual so the 1-D outer functions get a clean
            # gradient path independent of the cross-channel mixer.
            self.kst_gamma = nn.Parameter(torch.tensor(0.1))
        # (A) Post-superposition cross-band + spatial refinement (gated).
        self.post = (PostKSTRefine(bands, spatial=post_spatial)
                     if post_refine else None)
        # Cross-band polish (residual around identity), as in bc_usgs.
        self.gccm = GlobalColorMatrix(giv_dim, bands)
        self.lccm = LocalColorMatrixLR(dim, bands)
        # Bounded, bare-Parameter-gated correction. The pipeline's generic
        # re-init clobbers every Conv/Linear identity/zero init, so nothing
        # else guarantees a near-identity start; ``out_gate`` (a bare Parameter,
        # untouched by re-init) both soft-clamps the correction via tanh (it can
        # never blow up) and keeps the level close to identity at init.
        self.out_gate = nn.Parameter(torch.tensor(0.2))

    def forward(self, src: torch.Tensor,
                coarse_out: torch.Tensor = None) -> torch.Tensor:
        x_in = src if coarse_out is None else src + coarse_out

        giv = self.giv(x_in)                                 # [B, giv_dim]
        coords = self.channel_mix(x_in)                      # x [B, n, H, W]
        feat, illu_fea, _ = self.coeff_enc(x_in, giv)        # control features
        ctrl = torch.cat([feat, illu_fea], dim=1)            # hypernet input

        xi = self.core(coords, ctrl)                         # [B, Qo, H, W]
        if self.readout == 'kan':
            y = self.kan(xi)                                 # Phi_{q,o} grid
        else:
            u = self.phi(xi)                                 # Phi_q(xi_q) 1-D
            y = self.mixer(u, giv)                           # external mixer
            y = y + self.kst_gamma * u.sum(1, keepdim=True)  # pure-KST term
        if self.post is not None:
            y = self.post(y)                                 # (A) refine spectra
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
                 dim: int = 96, giv_dim: int = 64, phi_dim: int = 64,
                 phi_heads: int = 4, readout: str = 'kan', kan_basis: int = 8,
                 use_fourier: bool = True, use_checkpoint: bool = True,
                 upsample: str = 'bilinear', illu_dim: int = 32,
                 head_spatial: bool = True, spectral_prior: bool = True,
                 post_refine: bool = True, post_spatial: bool = True,
                 fine_refine: bool = True):
        super().__init__()
        assert upsample in ('wavelet', 'bilinear')
        self.upsample = upsample
        Qs = [Qo] * 3 if isinstance(Qo, int) else list(Qo)
        Ms = [M] * 3 if isinstance(M, int) else list(M)
        Ds = [dim] * 3 if isinstance(dim, int) else list(dim)

        def _level(idx: int) -> KSTSagfLevel:
            return KSTSagfLevel(
                bands=bands, n=n, Qo=Qs[idx], M=Ms[idx], dim=Ds[idx],
                giv_dim=giv_dim, blocks=depths[idx], phi_dim=phi_dim,
                phi_heads=phi_heads, readout=readout, kan_basis=kan_basis,
                use_fourier=use_fourier, use_checkpoint=use_checkpoint,
                illu_dim=illu_dim, head_spatial=head_spatial,
                spectral_prior=spectral_prior, post_refine=post_refine,
                post_spatial=post_spatial)

        self.coarse = _level(0)
        self.mid = _level(1)
        self.fine = _level(2)
        # (E) Full-res residual refiner on the finest output only.
        self.refine = FineRefiner(bands) if fine_refine else None
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
        if self.refine is not None:
            y = self.refine(y)                               # (E) full-res detail
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
        if self.refine is not None:
            y = self.refine(y)                               # (E) full-res detail
        return y, y2, y4

    def forward(self, src: torch.Tensor):
        if self.upsample == 'wavelet':
            y, y2, y4 = self._forward_wavelet(src)
        else:
            y, y2, y4 = self._forward_bilinear(src)
        if self.training:
            return y, y2, y4
        return y
