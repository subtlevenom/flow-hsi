import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Kept from v17 unchanged ──────────────────────────────────────────

class GELU(nn.Module):
    def forward(self, x): return F.gelu(x)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.LayerNorm(dim)
    def forward(self, x):
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.body(x)
            return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.body(x)

class Advanced_GFFN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.project_in = nn.Conv2d(in_dim, out_dim * 2, 1)
        self.dwconv_3x3 = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim)
        self.dwconv_5x5 = nn.Conv2d(out_dim, out_dim, 5, padding=2, groups=out_dim)
        hd = max(1, out_dim // 4)
        self.spectral_calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim * 2, hd, 1), GELU(),
            nn.Conv2d(hd, out_dim * 2, 1), nn.Sigmoid())
        self.project_out = nn.Conv2d(out_dim, out_dim, 1)
    def forward(self, x):
        c = self.project_in(x)
        c = c * self.spectral_calibration(c)
        x1, x2 = c.chunk(2, dim=1)
        return self.project_out(
            self.dwconv_3x3(x1) * torch.sigmoid(self.dwconv_5x5(x2)))

class RecursiveFractalChi(nn.Module):
    # Kept exactly from v17 — theoretically sound
    def __init__(self, x_dim, psi_dim, out_dim):
        super().__init__()
        self.x_norm = LayerNorm(x_dim)
        cd = psi_dim + x_dim
        self.dw1 = nn.Conv2d(cd, cd, 3, padding=1, groups=cd)
        self.dw2 = nn.Conv2d(cd, cd, 5, padding=2, groups=cd)
        self.gate1 = nn.Sequential(nn.Conv2d(cd, cd, 1), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Conv2d(cd, cd, 1), nn.Sigmoid())
        self.proj_out = nn.Conv2d(cd, out_dim, 1)
    def forward(self, psi, x):
        x = self.x_norm(x)
        pc = torch.cat([psi, x], dim=1)
        fh = self.dw2(pc) * self.gate2(pc)
        fm = self.dw1(pc + fh) * self.gate1(fh)
        return self.proj_out(fm), self.gate1(fh)

class LaplacianGatedFusion(nn.Module):
    # Kept from v17 — works well
    def __init__(self, in_c, out_c):
        super().__init__()
        self.lap = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c)
        self.gate = nn.Sequential(
            nn.Conv2d(in_c + out_c + in_c, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_c, 1))
        self.refine = Advanced_GFFN(in_c + out_c, out_c)
    def forward(self, x_orig, usgs_out, illu_map):
        x_detail = x_orig - self.lap(x_orig)
        gate = torch.sigmoid(
            self.gate(torch.cat([x_orig, usgs_out, illu_map], dim=1)))
        blended = usgs_out * gate + (x_orig + x_detail) * (1 - gate)
        return self.refine(torch.cat([blended, x_orig], dim=1))


# ── IDEA A: HAIR-style DAC → GIV ────────────────────────────────────
# Replaces SpectralOrchestrator
# Uses mean+variance at 3 spatial scales → richer scene encoding

class DegradationAwareConditioner(nn.Module):
    """
    Inspired by HAIR (arXiv:2408.08091) DAC.
    Extracts multi-scale mean+variance statistics from the input
    to produce a Global Information Vector (GIV) that captures:
    - Global white balance (1x1 stats)
    - Regional illumination gradients (4x4 stats)
    - Local color variance (8x8 stats)
    This is strictly richer than v17's AdaptiveAvgPool(1)+Sigmoid.
    """
    def __init__(self, in_c, giv_dim):
        super().__init__()
        # 3 scales × 2 statistics (mean, var) × in_c channels
        stat_dim = in_c * 2 * (1 + 16 + 64)
        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, giv_dim * 2),
            nn.GELU(),
            nn.Linear(giv_dim * 2, giv_dim),
            nn.LayerNorm(giv_dim),   # stabilizes GIV magnitude
        )

    def _scale_stats(self, x, size):
        mu = F.adaptive_avg_pool2d(x, size).flatten(1)
        # variance: E[x^2] - E[x]^2 at this scale
        mu2 = F.adaptive_avg_pool2d(x ** 2, size).flatten(1)
        var = (mu2 - mu ** 2).clamp(min=0)
        return torch.cat([mu, var], dim=1)

    def forward(self, x):
        s1 = self._scale_stats(x, 1)   # [B, in_c*2*1]
        s4 = self._scale_stats(x, 4)   # [B, in_c*2*16]
        s8 = self._scale_stats(x, 8)   # [B, in_c*2*64]
        return self.mlp(torch.cat([s1, s4, s8], dim=1))  # [B, giv_dim]


# ── IDEA A (cont): FiLM injection ───────────────────────────────────
# GIV → (gamma, beta) per spatial feature channel
# Strictly better than concatenation: no channel budget wasted

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    giv [B, giv_dim] → gamma, beta [B, feat_dim, 1, 1]
    Applied as: out = feat * (1 + gamma) + beta
    The residual form (1 + gamma) ensures identity at init.
    """
    def __init__(self, giv_dim, feat_dim):
        super().__init__()
        self.proj = nn.Linear(giv_dim, feat_dim * 2)
        # Init to near-zero so FiLM starts as identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat, giv):
        gb = self.proj(giv).view(giv.size(0), -1, 1, 1)
        gamma, beta = gb.chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta


# ── IDEA B+C: Full-Resolution Multi-Scale Encoder ───────────────────
# Replaces Encoder2D entirely.
# No DWT, no striding. Parallel dilated branches (cmKAN MCA concept)
# + SAFMN-style channel mixing + FiLM from GIV.

class FullResEncoder(nn.Module):
    """
    Key design principles:
    1. NO downsampling — every pixel keeps its spatial identity
       through the entire encoding path (cmKAN principle).
    2. Multi-scale context via dilation (rates 1, 2, 4) instead
       of spatial pooling — captures local edges, mid-range
       illumination gradients, and global WB context simultaneously.
    3. FiLM injection of GIV after spatial mixing — the global
       scene descriptor modulates the spatial feature map
       channel-wise (HAIR HSN principle).
    4. Illumination map estimated at full resolution, used
       downstream in sigma_boost (preserved from v17).
    """
    def __init__(self, in_c, feat_dim, giv_dim):
        super().__init__()

        # Stem: project to feature space
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, feat_dim, 3, padding=1),
            nn.GroupNorm(min(4, feat_dim), feat_dim),
            nn.GELU(),
        )

        # Parallel dilated branches — full resolution, no stride
        # depthwise to keep params low while expanding receptive field
        self.d1 = nn.Conv2d(feat_dim, feat_dim, 3,
                            padding=1,  dilation=1, groups=feat_dim)
        self.d2 = nn.Conv2d(feat_dim, feat_dim, 3,
                            padding=2,  dilation=2, groups=feat_dim)
        self.d4 = nn.Conv2d(feat_dim, feat_dim, 3,
                            padding=4,  dilation=4, groups=feat_dim)

        # SAFMN-style channel mixer: fuse multi-scale branches
        # pointwise → GELU → pointwise (no spatial compression)
        self.channel_mix = nn.Sequential(
            nn.Conv2d(feat_dim * 3, feat_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(feat_dim * 2, feat_dim, 1),
            nn.GroupNorm(min(4, feat_dim), feat_dim),
        )

        # FiLM: inject GIV into spatial feature map
        self.film = FiLM(giv_dim, feat_dim)

        # Illumination estimator (full-res, preserved from v17 logic)
        self.illu_conv1 = nn.Conv2d(in_c + 1, 16, 1)
        self.illu_dw    = nn.Conv2d(16, 16, 5, padding=4,
                                    dilation=2, groups=16)
        self.illu_conv2 = nn.Conv2d(16, in_c, 1)

    def forward(self, x, giv):
        # Illumination map (full resolution)
        illu_in  = torch.cat([x, x.mean(1, keepdim=True)], dim=1)
        illu_fea = self.illu_dw(self.illu_conv1(illu_in))
        illu_map = torch.exp(
            torch.clamp(self.illu_conv2(illu_fea), -2, 2))

        # Full-resolution multi-scale feature extraction
        f  = self.stem(x)
        b1 = F.gelu(self.d1(f))
        b2 = F.gelu(self.d2(f))
        b4 = F.gelu(self.d4(f))

        # Fuse branches (SAFMN channel mixing)
        feat = self.channel_mix(torch.cat([b1, b2, b4], dim=1))

        # Inject global scene context via FiLM
        feat = self.film(feat, giv)

        return feat, illu_fea, illu_map


# ── Upgraded Expert Head ─────────────────────────────────────────────
# Replaces HyperExpertHead + LightMSAB.
# SAFMN-style: channel mix + depthwise spatial mix, no striding.
# Per-expert FiLM so each expert has its own scene-conditioned view.

class SpatialExpertHead(nn.Module):
    """
    SAFMN-inspired full-resolution parameter predictor.

    Unlike v17's LightMSAB (which uses strided Q/K projections),
    this uses:
    - Channel mixing (1x1 conv): captures cross-channel spectral
      relations (R↔Y↔B coupling in RYYB)
    - Depthwise spatial mixing (3x3 dw): captures local context
    - Per-expert FiLM: each expert is independently conditioned
      on the global scene descriptor GIV

    No striding → no spatial information loss in parameter prediction.
    """
class SpatialExpertHead(nn.Module):
    def __init__(self, feat_dim, illu_dim, out_c, Q, giv_dim):
        super().__init__()
        all_params = out_c * 3 * Q
        in_dim = feat_dim + illu_dim  # 64 — not divisible by out_c=3

        # Round up to nearest multiple of out_c so groups= out_c is valid
        # 64 → 66  (66 % 3 == 0)
        self.proj_dim = math.ceil(in_dim / out_c) * out_c  # 66

        self.film   = FiLM(giv_dim, in_dim)
        self.ch_mix = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(in_dim * 2, in_dim, 1),
        )
        self.sp_mix = nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim)

        self.norm = LayerNorm(in_dim)

        # Adapter: in_dim (64) → proj_dim (66), then grouped conv
        self.pre_proj = nn.Conv2d(in_dim, self.proj_dim, 1, bias=False)
        self.proj     = nn.Conv2d(self.proj_dim, all_params, 1, groups=out_c)
        self.skip     = nn.Conv2d(in_dim, all_params, 1)   # skip stays on in_dim
        self.gamma    = nn.Parameter(torch.ones(1, all_params, 1, 1) * 0.1)

    def forward(self, feat, illu_fea, giv):
        x = torch.cat([feat, illu_fea], dim=1)   # [B, 64, H, W]
        x = self.film(x, giv)
        x = x + self.ch_mix(x) + self.sp_mix(x)
        normed = self.norm(x)
        return self.skip(x) + self.proj(self.pre_proj(normed)) * self.gamma


# ── HGSABlock v18 ────────────────────────────────────────────────────

class HGSABlock_v18(nn.Module):
    """
    Core USGS manifold block. Key changes vs v17:

    1. xi = direct pixel input (no BasisAttention compression).
       Pixel identity is fully preserved into the Gaussian kernel.

    2. mu_base initialized as evenly-spaced grid [0.1..0.9].
       Experts predict offsets from this grid, not absolute values.
       This prevents mode collapse where all experts predict
       the same mu.

    3. Each expert independently conditioned by GIV via FiLM
       inside SpatialExpertHead.

    4. sigma_boost from illu_map preserved (it works in v17).

    5. spectral_calibrator and RecursiveFractalChi preserved.
    """
    def __init__(self, in_c=3, out_c=3, Q=8, M=4,
                 feat_dim=48, illu_dim=16, giv_dim=64):
        super().__init__()
        self.Q, self.M, self.out_c = Q, M, out_c

        self.expert_heads = nn.ModuleList([
            SpatialExpertHead(feat_dim, illu_dim, out_c, Q, giv_dim)
            for _ in range(M)
        ])

        # Evenly-spaced spectral grid initialization
        # Much better than random: experts start covering the full [0,1] range
        mu_grid = torch.linspace(0.1, 0.9, Q).view(1, 1, Q, 1, 1)
        self.mu_base  = nn.Parameter(
            mu_grid.expand(1, out_c, Q, 1, 1).clone())
        self.mu_scale = nn.Parameter(
            torch.ones(1, out_c, Q, 1, 1) * 4.5)
        self.w_init   = nn.Parameter(
            0.1 * torch.randn(1, out_c, Q, 1, 1))
        self.sigma_init = nn.Parameter(
            torch.ones(M, out_c, Q) * 0.2)

        sq = out_c * Q
        self.spectral_calibrator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(sq, max(1, sq // 4), 1), GELU(),
            nn.Conv2d(max(1, sq // 4), sq, 1), nn.Sigmoid())

        self.chi_net = RecursiveFractalChi(in_c, sq, out_c)

    def forward(self, x, feat, illu_fea, illu_map, giv):
        B, C, H, W = x.shape

        # Direct pixel input — no compression, full identity preserved
        xi = x.unsqueeze(2).expand(-1, -1, self.Q, -1, -1)

        sigma_boost = torch.clamp(
            1.0 / (illu_map.mean(1, keepdim=True) + 1e-4),
            1.0, 2.5).unsqueeze(2)

        psi_total = torch.zeros(
            B, self.out_c, self.Q, H, W, device=x.device)

        for i in range(self.M):
            # Full-resolution parameter map, per-expert GIV conditioning
            p_e = self.expert_heads[i](feat, illu_fea, giv).view(
                B, self.out_c, 3, self.Q, H, W)

            w     = self.w_init + p_e[:, :, 0]
            # mu: offset from evenly-spaced grid
            mu    = (torch.tanh(self.mu_base + p_e[:, :, 1])
                     * F.softplus(self.mu_scale) + 0.5)
            s_b   = self.sigma_init[i].view(1, self.out_c, self.Q, 1, 1)
            sigma = (F.softplus(s_b + p_e[:, :, 2]) + 0.01) * sigma_boost

            g = torch.exp(-0.5 * ((xi - mu) / sigma).pow(2))
            psi_total = psi_total + w * g

        psi_flat = psi_total.view(B, self.out_c * self.Q, H, W)
        psi_flat = psi_flat * self.spectral_calibrator(psi_flat)
        out, _   = self.chi_net(psi_flat, x)
        return out, psi_flat


# ── IDEA #2: GIV-conditioned global color matrix ─────────────────────
# The USGS core is strictly per-channel (xi = x.unsqueeze(2)), so chroma
# cross-talk (target R depends on source R, G, B) otherwise has to be
# reconstructed entirely by chi_net/fusion. This module provides an
# explicit, GIV-conditioned linear color transform (3x3 CCM + bias),
# parameterized as a residual around identity so it starts as a no-op.

class GlobalColorMatrix(nn.Module):
    def __init__(self, giv_dim, c=3):
        super().__init__()
        self.c = c
        self.proj = nn.Linear(giv_dim, c * c + c)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)   # identity CCM, zero bias at init

    def forward(self, x, giv):
        B = x.size(0)
        p = self.proj(giv)
        M = p[:, :self.c * self.c].view(B, self.c, self.c)
        b = p[:, self.c * self.c:].view(B, self.c, 1, 1)
        M = torch.eye(self.c, device=x.device, dtype=x.dtype).unsqueeze(0) + M
        return torch.einsum('boc,bchw->bohw', M, x) + b


# ── Top-level HGSA_v18 ───────────────────────────────────────────────

class HGSA_v18(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, Q=8, M=4):
        super().__init__()
        FEAT_DIM = 48
        ILLU_DIM = 16
        GIV_DIM  = 64

        # HAIR-style global scene conditioner
        self.conditioner = DegradationAwareConditioner(in_channels, GIV_DIM)

        # GIV-conditioned global color matrix — explicit cross-channel
        # (WB/CCM) coupling that the per-channel USGS core cannot model
        self.ccm = GlobalColorMatrix(GIV_DIM, in_channels)

        # Full-resolution encoder, no DWT bottleneck
        self.encoder = FullResEncoder(in_channels, FEAT_DIM, GIV_DIM)

        # USGS manifold block
        self.usgs = HGSABlock_v18(
            in_channels, out_channels, Q, M, FEAT_DIM, ILLU_DIM, GIV_DIM)

        # Laplacian-gated texture fusion (preserved from v17)
        self.fusion = LaplacianGatedFusion(in_channels, out_channels)

        # Aux head for intermediate supervision
        self.aux_proj = nn.Conv2d(Q * out_channels, out_channels, 1)

    def forward(self, x):
        # 1. Global scene descriptor (HAIR DAC concept)
        giv = self.conditioner(x)

        # 1b. Explicit linear cross-channel color correction (WB/CCM prior).
        #     Handles the linear color component; the USGS manifold then
        #     refines the non-linear residual on top of the corrected base.
        x_wb = self.ccm(x, giv)

        # 2. Full-resolution spatial features + illumination (from raw input)
        feat, illu_fea, illu_map = self.encoder(x, giv)

        # 3. USGS manifold: spatially-adaptive Gaussian superposition
        usgs_out, psi_raw = self.usgs(
            x_wb, feat, illu_fea, illu_map, giv)

        # 4. Laplacian-gated texture fusion (against color-corrected base)
        out = self.fusion(x_wb, usgs_out, illu_map)

        if self.training:
            return out, x_wb + self.aux_proj(psi_raw)
        return out
