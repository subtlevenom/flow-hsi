import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ─── 1. BASIC UTILITIES ──────────────────────────────────────────────

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

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
    def __init__(self, in_c, feat_dim, giv_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, feat_dim, 3, padding=1),
            nn.GroupNorm(4, feat_dim), GELU()
        )
        self.d1 = nn.Conv2d(feat_dim, feat_dim, 3, padding=1, dilation=1, groups=feat_dim)
        self.d2 = nn.Conv2d(feat_dim, feat_dim, 3, padding=2, dilation=2, groups=feat_dim)
        self.d4 = nn.Conv2d(feat_dim, feat_dim, 3, padding=4, dilation=4, groups=feat_dim)
        
        self.fuse = nn.Sequential(nn.Conv2d(feat_dim * 3, feat_dim, 1), nn.GroupNorm(4, feat_dim))
        self.film = FiLM(giv_dim, feat_dim)
        
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

# ─── 3. EOT-USGS CORE ────────────────────────────────────────────────

class TransportGainHead(nn.Module):
    """
    Predicts Local Transport Mappings Tq(x).
    Uses a hypernetwork approach to generate coefficients for:
    T(x) = mu + K1(x-mu) + K2(x-mu)^2
    """
    def __init__(self, feat_dim, Q, giv_dim):
        super().__init__()
        self.Q = Q
        # Parameters: 9 (K1) + 9 (K2) + 3 (mu offset) = 21
        self.params_per_q = 21 
        self.film = FiLM(giv_dim, feat_dim)
        
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            nn.Conv2d(feat_dim, self.params_per_q * Q, 1)
        )
        self.global_gate = nn.Sequential(
            nn.Linear(giv_dim, self.params_per_q * Q),
            nn.Sigmoid()
        )

    def forward(self, feat, giv):
        local_p = self.net(self.film(feat, giv))
        global_p = self.global_gate(giv).view(giv.size(0), -1, 1, 1)
        p = local_p * global_p
        return p.view(p.shape[0], self.Q, self.params_per_q, p.shape[2], p.shape[3])

class SAGFLayer(nn.Module):
    """
    Internal Sum: Formulates the Gibbs Kernel Kq(x, y) = sum( am * exp(-||y-mu||^2 / 2sigma^2) ).
    """
    def __init__(self, feat_dim, Q, M, giv_dim):
        super().__init__()
        self.Q, self.M = Q, M
        # Parameters per Gaussian: 1 (a) + 3 (mu) + 1 (sigma) = 5
        self.params_per_m = 5
        self.hyper = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            nn.Conv2d(feat_dim, Q * M * self.params_per_m, 1)
        )
        self.film = FiLM(giv_dim, feat_dim)

    def forward(self, x, feat, giv):
        B, C, H, W = x.shape
        p = self.hyper(self.film(feat, giv))
        p = p.view(B, self.Q, self.M, self.params_per_m, H, W)
        
        # Extract components
        a = torch.sigmoid(p[:, :, :, 0:1])
        mu = torch.sigmoid(p[:, :, :, 1:4])
        sigma = F.softplus(p[:, :, :, 4:5]) + 1e-4
        
        # x: [B, 1, 1, 3, H, W], mu: [B, Q, M, 3, H, W]
        x_expanded = x.unsqueeze(1).unsqueeze(2) 
        dist_sq = torch.sum((x_expanded - mu)**2, dim=3, keepdim=True)
        
        # Gibbs Kernel component
        kernels = a * torch.exp(-0.5 * dist_sq / (sigma**2))
        return torch.sum(kernels, dim=2) # [B, Q, 1, H, W]

class EOT_USGS_Block(nn.Module):
    """
    Barycentric Projection T(x) = sum( pi_q * Tq(x) ) / sum( pi_q ).
    """
    def __init__(self, in_c=3, out_c=3, Q=8, M=4, feat_dim=64, giv_dim=64):
        super().__init__()
        self.Q, self.M = Q, M
        self.gibbs_layer = SAGFLayer(feat_dim, Q, M, giv_dim)
        self.transport_head = TransportGainHead(feat_dim, Q, giv_dim)
        
        # Potential potential psi(g)
        self.phi_gate = nn.Sequential(
            nn.Linear(giv_dim, Q),
            nn.Softmax(dim=1)
        )

    def forward(self, x_img, feat, giv, illu_map):
        B, C, H, W = x_img.shape
        
        # 1. Compute Gibbs Kernels Kq(x, y)
        K = self.gibbs_layer(x_img, feat, giv) # [B, Q, 1, H, W]
        
        # 2. External potential coupling (conjugation)
        psi = self.phi_gate(giv).view(B, self.Q, 1, 1, 1)
        pi_q = K * psi # The transport plan component
        
        # 3. Compute Local Transport Mappings Tq(x)
        p = self.transport_head(feat, giv)
        k1_raw = p[:, :, 0:9].view(B, self.Q, 3, 3, H, W)
        k2_raw = p[:, :, 9:18].view(B, self.Q, 3, 3, H, W)
        mu_loc = torch.sigmoid(p[:, :, 18:21])
        
        # Functional expansion
        eye = torch.eye(3, device=x_img.device).view(1, 1, 3, 3, 1, 1)
        k1 = eye + 0.1 * torch.tanh(k1_raw)
        k2 = 0.02 * torch.tanh(k2_raw)
        
        # Tq(x) calculation
        diff = (x_img.unsqueeze(1) - mu_loc).permute(0, 1, 3, 4, 2).unsqueeze(-1)
        k1_mat = k1.permute(0, 1, 4, 5, 2, 3) 
        k2_mat = k2.permute(0, 1, 4, 5, 2, 3)
        
        trans = torch.matmul(k1_mat, diff) + torch.matmul(k2_mat, diff**2)
        T_q = mu_loc + trans.squeeze(-1).permute(0, 1, 4, 2, 3) # [B, Q, 3, H, W]
        
        # 4. Barycentric Projection
        num = torch.sum(pi_q * T_q, dim=1)
        den = torch.sum(pi_q, dim=1) + 1e-6
        
        return num / den

# ─── 4. REFINEMENT & TOP-LEVEL ───────────────────────────────────────

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
        self.refine = nn.Sequential(
            nn.Conv2d(in_c + out_c, out_c * 2, 1),
            # Changed groups from 4 to 2 (6 is divisible by 2)
            nn.GroupNorm(2, out_c * 2), 
            GELU(),
            nn.Conv2d(out_c * 2, out_c, 3, padding=1)
        )

    def forward(self, x_orig, transported, illu_map):
        x_detail = x_orig - self.lap(x_orig)
        g = self.gate(torch.cat([x_orig, transported, illu_map], dim=1))
        blended = transported * g + (x_orig + x_detail) * (1 - g)
        return self.refine(torch.cat([blended, x_orig], dim=1))

class HGSA_GEOT_v20(nn.Module):
    """
    Mathematical synthesis of USGS and EOT.
    Uses Barycentric Projection of local transport maps weighted by the Gibbs plan.
    """
    def __init__(self, in_channels=3, out_channels=3, Q=12, M=6):
        super().__init__()
        GIV_DIM, FEAT_DIM = 64, 64
        
        self.conditioner = DegradationAwareConditioner(in_channels, GIV_DIM)
        self.encoder = FullResEncoder(in_channels, FEAT_DIM, GIV_DIM)
        self.eot_usgs = EOT_USGS_Block(in_channels, out_channels, Q, M, FEAT_DIM, GIV_DIM)
        self.fusion = LaplacianGatedFusion(in_channels, out_channels)

    def forward(self, src):
        giv = self.conditioner(src)
        feat, illu_map = self.encoder(src, giv)
        
        # Step 1: EOT-USGS Synthesis (Gibbs Kernel + Transport Map)
        transported_x = self.eot_usgs(src, feat, giv, illu_map)
        
        # Step 2: Refinement
        final_out = self.fusion(src, transported_x, illu_map)
        
        if self.training:
            return {'res': (final_out, transported_x)}
        return {'res': final_out}