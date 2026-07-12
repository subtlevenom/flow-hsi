
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ─── 1. BASIC UTILITIES ──────────────────────────────────────────────

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

class FiLM(nn.Module):
    """Feature-wise Linear Modulation for global conditioning."""
    def __init__(self, giv_dim, feat_dim):
        super().__init__()
        self.proj = nn.Linear(giv_dim, feat_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat, giv):
        gb = self.proj(giv).view(giv.size(0), -1, 1, 1)
        gamma, beta = gb.chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta

# ─── 2. CONDITIONING & ENCODING ──────────────────────────────────────

class DegradationAwareConditioner(nn.Module):
    """Extracts Global Information Vector (GIV) via multi-scale stats."""
    def __init__(self, in_c, giv_dim):
        super().__init__()
        stat_dim = in_c * 2 * (1 + 16 + 64)
        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, giv_dim * 2),
            GELU(),
            nn.Linear(giv_dim * 2, giv_dim),
            nn.LayerNorm(giv_dim)
        )

    def _scale_stats(self, x, size):
        mu = F.adaptive_avg_pool2d(x, size).flatten(1)
        mu2 = F.adaptive_avg_pool2d(x ** 2, size).flatten(1)
        var = (mu2 - mu ** 2).clamp(min=0)
        return torch.cat([mu, var], dim=1)

    def forward(self, x):
        s1 = self._scale_stats(x, 1)
        s4 = self._scale_stats(x, 4)
        s8 = self._scale_stats(x, 8)
        return self.mlp(torch.cat([s1, s4, s8], dim=1))

class FullResEncoder(nn.Module):
    """Multi-scale dilated encoder with FiLM injection."""
    def __init__(self, in_c, feat_dim, giv_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, feat_dim, 3, padding=1),
            nn.GroupNorm(4, feat_dim), GELU()
        )
        self.d1 = nn.Conv2d(feat_dim, feat_dim, 3, padding=1, dilation=1, groups=feat_dim)
        self.d2 = nn.Conv2d(feat_dim, feat_dim, 3, padding=2, dilation=2, groups=feat_dim)
        self.d4 = nn.Conv2d(feat_dim, feat_dim, 3, padding=4, dilation=4, groups=feat_dim)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 3, feat_dim, 1),
            nn.GroupNorm(4, feat_dim)
        )
        self.film = FiLM(giv_dim, feat_dim)
        
        # Illumination Estimator
        self.illu_conv = nn.Sequential(
            nn.Conv2d(in_c + 1, 16, 1),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, giv):
        illu_map = self.illu_conv(torch.cat([x, x.mean(1, keepdim=True)], dim=1))
        f = self.stem(x)
        feat = self.fuse(torch.cat([self.d1(f), self.d2(f), self.d4(f)], dim=1))
        return self.film(feat, giv), illu_map

# ─── 3. GEOT-USGS CORE ───────────────────────────────────────────────

class TransportGainHead(nn.Module):
    """Predicts 3x3 Gain Matrix (kappa), Mean (mu), and Weight (w)."""
    def __init__(self, feat_dim, out_c, Q, giv_dim):
        super().__init__()
        self.Q = Q
        self.params_per_q = (out_c * out_c) + out_c + 1 
        self.film = FiLM(giv_dim, feat_dim)
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            nn.Conv2d(feat_dim, self.params_per_q * Q, 1)
        )

    def forward(self, feat, giv):
        p = self.net(self.film(feat, giv))
        return p.view(p.shape[0], self.Q, self.params_per_q, p.shape[2], p.shape[3])

class GEOT_USGS_Block(nn.Module):
    """Synthesis of USGS and Gaussian Entropic Optimal Transport."""
    def __init__(self, in_c=3, out_c=3, Q=8, M=4, feat_dim=64, giv_dim=64):
        super().__init__()
        self.Q, self.M, self.out_c = Q, M, out_c
        self.experts = nn.ModuleList([TransportGainHead(feat_dim, out_c, Q, giv_dim) for _ in range(M)])
        
        mu_grid = torch.linspace(0.1, 0.9, Q).view(1, Q, 1, 1, 1)
        self.mu_anchor = nn.Parameter(mu_grid.expand(1, Q, out_c, 1, 1).clone())
        self.sigma_base = nn.Parameter(torch.ones(1, Q, 1, 1, 1) * 0.2)

    def forward(self, x_img, feat, giv, illu_map):
        B, C, H, W = x_img.shape
        x_flat = x_img.unsqueeze(1).expand(-1, self.Q, -1, -1, -1)
        tau = torch.clamp(1.0 / (illu_map + 1e-4), 1.0, 2.0).unsqueeze(1)

        num = torch.zeros_like(x_img)
        den = torch.zeros(B, 1, H, W, device=x_img.device) + 1e-6

        for m in range(self.M):
            p = self.experts[m](feat, giv)
            kappa_raw = p[:, :, 0:9].view(B, self.Q, 3, 3, H, W)
            mu_off    = p[:, :, 9:12]
            logit_w   = p[:, :, 12:13]
            
            mu = torch.sigmoid(self.mu_anchor + mu_off)
            eye = torch.eye(3, device=x_img.device).view(1, 1, 3, 3, 1, 1)
            kappa = eye + 0.1 * torch.tanh(kappa_raw)
            
            sigma = F.softplus(self.sigma_base) * tau
            dist_sq = torch.sum((x_flat - mu)**2, dim=2, keepdim=True)
            kernel = torch.exp(-0.5 * dist_sq / (sigma**2))
            
            # T(x) = mu + kappa(x - mu)
            diff = (x_flat - mu).permute(0, 1, 3, 4, 2).unsqueeze(-1)
            k_mat = kappa.permute(0, 1, 4, 5, 2, 3)
            trans = torch.matmul(k_mat, diff).squeeze(-1).permute(0, 1, 4, 2, 3)
            t_map = mu + trans
            
            w = torch.softmax(logit_w, dim=1) * kernel
            num = num + torch.sum(w * t_map, dim=1)
            den = den + torch.sum(w, dim=1)

        return num / den

# ─── 4. REFINEMENT & FUSION ──────────────────────────────────────────

class Advanced_GFFN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.project_in = nn.Conv2d(in_dim, out_dim * 2, 1)
        self.dwconv = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim)
        self.project_out = nn.Conv2d(out_dim, out_dim, 1)
    def forward(self, x):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        return self.project_out(self.dwconv(x1) * torch.sigmoid(x2))

class LaplacianGatedFusion(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.lap = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c)
        self.gate = nn.Sequential(
            nn.Conv2d(in_c + out_c + 1, 16, 3, padding=1),
            GELU(),
            nn.Conv2d(16, out_c, 1),
            nn.Sigmoid()
        )
        self.refine = Advanced_GFFN(in_c + out_c, out_c)

    def forward(self, x_orig, transported, illu_map):
        x_detail = x_orig - self.lap(x_orig)
        g = self.gate(torch.cat([x_orig, transported, illu_map], dim=1))
        blended = transported * g + (x_orig + x_detail) * (1 - g)
        return self.refine(torch.cat([blended, x_orig], dim=1))

# ─── 5. TOP-LEVEL MODEL ──────────────────────────────────────────────

class HGSA_GEOT_v19(nn.Module):
    """
    USGS & GEOT: Gaussian Transport Synthesis for Color Matching.
    """
    def __init__(self, in_channels=3, out_channels=3, Q=8, M=4):
        super().__init__()
        GIV_DIM = 64
        FEAT_DIM = 64
        
        self.conditioner = DegradationAwareConditioner(in_channels, GIV_DIM)
        self.encoder = FullResEncoder(in_channels, FEAT_DIM, GIV_DIM)
        self.geot_transport = GEOT_USGS_Block(in_channels, out_channels, Q, M, FEAT_DIM, GIV_DIM)
        self.fusion = LaplacianGatedFusion(in_channels, out_channels)

    def forward(self, src):
        giv = self.conditioner(src)
        feat, illu_map = self.encoder(src, giv)
        
        # Step 1: Perform Optimal Transport in color space
        transported_x = self.geot_transport(src, feat, giv, illu_map)
        
        # Step 2: Refine texture and edges via Laplacian fusion
        final_out = self.fusion(src, transported_x, illu_map)
        
        if self.training:
            return {'res': (final_out, transported_x)}
        return {'res': final_out}