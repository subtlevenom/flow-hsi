import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --- Base Components ---


class GELU(nn.Module):

    def forward(self, x):
        return F.gelu(x)


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


class DWTForward(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        # Space to Channel: (B, C, H, W) -> (B, 4C, H/2, W/2)
        x01, x23 = x[:, :, 0::2, :] / 2, x[:, :, 1::2, :] / 2
        x_ll = x01[:, :, :, 0::2] + x23[:, :, :,
                                        0::2] + x01[:, :, :,
                                                    1::2] + x23[:, :, :, 1::2]
        x_lh = -x01[:, :, :, 0::2] - x23[:, :, :,
                                         0::2] + x01[:, :, :,
                                                     1::2] + x23[:, :, :, 1::2]
        x_hl = -x01[:, :, :, 0::2] + x23[:, :, :,
                                         0::2] - x01[:, :, :,
                                                     1::2] + x23[:, :, :, 1::2]
        x_hh = x01[:, :, :, 0::2] - x23[:, :, :,
                                        0::2] - x01[:, :, :,
                                                    1::2] + x23[:, :, :, 1::2]
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)


# --- v17 Specific: Texture Preservation & Color Calibration ---


class Advanced_GFFN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.project_in = nn.Conv2d(in_dim, out_dim * 2, 1)
        self.dwconv_3x3 = nn.Conv2d(out_dim,
                                    out_dim,
                                    3,
                                    padding=1,
                                    groups=out_dim)
        self.dwconv_5x5 = nn.Conv2d(out_dim,
                                    out_dim,
                                    5,
                                    padding=2,
                                    groups=out_dim)
        hidden_dim = max(1, out_dim // 4)
        self.spectral_calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim * 2, hidden_dim, 1),
            GELU(),
            nn.Conv2d(hidden_dim, out_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.project_out = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, x):
        combined = self.project_in(x)
        combined = combined * self.spectral_calibration(combined)
        x1, x2 = combined.chunk(2, dim=1)
        return self.project_out(
            self.dwconv_3x3(x1) * torch.sigmoid(self.dwconv_5x5(x2)))


class LaplacianGatedFusion(nn.Module):
    """
    v17 Improvement: Bridge the dE gap by selectively fusing 
    original high-frequency textures back into the denoised output.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(16, out_channels, 1),
            nn.Sigmoid())
        self.refine = Advanced_GFFN(in_channels + out_channels, out_channels)

    def forward(self, x_orig, usgs_out, illu_map):
        # Decide where the 'popcorn' texture needs to be preserved
        gate = self.gate_net(torch.cat([x_orig, usgs_out], dim=1))

        # Modulate by illumination: Trust original more in bright/textured areas
        effective_gate = gate * (1.0 - illu_map * 0.4)
        blended = x_orig * (1.0 - effective_gate) + usgs_out * effective_gate

        return self.refine(torch.cat([blended, x_orig], dim=1))


# --- USGS Core Components ---


class RGB_IlluminationEstimator(nn.Module):

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, 1)
        self.depth_conv = nn.Conv2d(n_fea_middle,
                                    n_fea_middle,
                                    5,
                                    padding=4,
                                    dilation=2,
                                    groups=n_fea_middle)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, 1)

    def forward(self, img):
        input_feat = torch.cat([img, img.mean(dim=1, keepdim=True)], dim=1)
        illu_fea = self.depth_conv(self.conv1(input_feat))
        illu_map = torch.exp(torch.clamp(self.conv2(illu_fea), -2, 2))
        return illu_fea, illu_map


class Attention(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature_a = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_v = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(dim,
                                dim,
                                3,
                                padding=1,
                                stride=2,
                                groups=dim,
                                bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, 3, padding=1, stride=2, bias=bias)
        self.a_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, stride=2, groups=dim, bias=bias),
            nn.Conv2d(dim, dim // 2, 1))

        self.v_proj = nn.Conv2d(dim, dim, 1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, illu_feat):
        b, c, h, w = x.shape
        q, k, a = self.q_proj(x), self.k_proj(x), self.a_proj(x)
        v = self.v_proj(x) * illu_feat

        q, k, a = [
            rearrange(t,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads) for t in (q, k, a)
        ]
        v = rearrange(v,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q, k, a = [F.normalize(t, dim=-1) for t in (q, k, a)]

        attn_a = (q @ a.transpose(-2, -1)) * self.temperature_a
        attn_k = (a @ k.transpose(-2, -1)) * self.temperature_v

        out = rearrange(attn_a.softmax(dim=-1) @ (attn_k.softmax(dim=-1) @ v),
                        'b head c (h w) -> b (head c) h w',
                        head=self.num_heads,
                        h=h,
                        w=w)
        return self.project_out(out)


class SpectralTransformerBlock(nn.Module):

    def __init__(self, in_channel, num_heads, bias):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm(in_channel), LayerNorm(in_channel)
        self.attn = Attention(in_channel, num_heads, bias)
        self.ffn = Advanced_GFFN(in_channel, in_channel)

    def forward(self, x, illu_feat):
        x = x + self.attn(self.norm1(x), illu_feat)
        return x + self.ffn(self.norm2(x))


class Encoder2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.estimator = RGB_IlluminationEstimator(16, in_channels + 1,
                                                   in_channels)
        self.down1 = DWTForward(in_channels)
        self.trans1 = SpectralTransformerBlock(12, 3, True)
        self.illu_down1 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(16, 12, 1))
        self.down2 = DWTForward(12)
        self.trans2 = SpectralTransformerBlock(48, 3, True)
        self.illu_down2 = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(16, 48, 1))
        self.up1, self.up2 = nn.Upsample(
            scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=4,
                                                          mode='bilinear')
        self.conv_out = nn.Sequential(
            LayerNorm(in_channels + 12 + 48),
            Advanced_GFFN(in_channels + 12 + 48, out_channels))

    def forward(self, x):
        illu_fea, illu_map = self.estimator(x)
        x_orig = x * illu_map
        x1 = self.trans1(self.down1(x_orig), self.illu_down1(illu_fea))
        x2 = self.trans2(self.down2(x1), self.illu_down2(illu_fea))
        out = torch.cat([x_orig, self.up1(x1), self.up2(x2)], dim=1)
        return self.conv_out(out), illu_fea, illu_map


# --- HGSABlock v17 Core ---


class SpectralOrchestrator(nn.Module):

    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.global_net = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(in_dim, cond_dim // 2, 1),
                                        GELU(),
                                        nn.Conv2d(cond_dim // 2, cond_dim, 1),
                                        nn.Sigmoid())

    def forward(self, x):
        return self.global_net(x)


class HyperExpertHead(nn.Module):

    def __init__(self, feat_dim, hidden_dim, out_channels, cond_dim):
        super().__init__()
        self.input_proj = nn.Conv2d(feat_dim + 16 + cond_dim, hidden_dim, 1)
        self.main_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.SiLU(), nn.Conv2d(hidden_dim, 3 * out_channels, 1))
        self.shortcut = nn.Conv2d(feat_dim + 16 + cond_dim, 3 * out_channels,
                                  1)
        self.gamma = nn.Parameter(torch.ones(1, 3 * out_channels, 1, 1) * 0.1)

    def forward(self, feat, illu_fea, v):
        v = v.expand(-1, -1, feat.size(2), feat.size(3))
        combined = torch.cat([feat, illu_fea, v], dim=1)
        return self.shortcut(combined) + self.main_branch(
            self.input_proj(combined)) * self.gamma


class BasisAttention(nn.Module):

    def __init__(self, x_dim, feat_dim, M, Q):
        super().__init__()
        self.num_concepts = 2 * M * Q
        self.basis_v = nn.Parameter(torch.randn(self.num_concepts, Q) * 0.02)
        self.q_proj = nn.Conv2d(x_dim, self.num_concepts, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(feat_dim, self.num_concepts, 1),
                                  nn.Sigmoid())
        self.tau = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, feat):
        B, _, H, W = x.shape
        q = self.q_proj(x) * self.gate(feat)
        q_flat = q.view(B, self.num_concepts, -1)
        attn = F.softmax(q_flat / self.tau, dim=1)
        v = self.basis_v.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
        return torch.bmm(v, attn).view(B, -1, H, W)


class DeepGatedChi(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.expansion = nn.Conv2d(in_dim, in_dim * 2, 1)
        self.dw_3x3 = nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim)
        self.dw_5x5 = nn.Conv2d(in_dim, in_dim, 5, padding=2, groups=in_dim)
        self.proj_out = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        x1, x2 = self.expansion(x).chunk(2, dim=1)
        gate = torch.sigmoid(self.dw_5x5(x2))
        res = self.proj_out(self.dw_3x3(x1) * gate)
        return res, gate


class HGSABlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, Q=7, M=5, feat_dim=48):
        super().__init__()
        self.cond_dim = 8
        self.M, self.Q, self.out_channels = M, Q, out_channels
        self.xi_net = BasisAttention(in_channels, feat_dim, M,
                                     out_channels * Q)
        self.orchestrator = SpectralOrchestrator(in_dim=in_channels,
                                                 cond_dim=M * self.cond_dim)
        self.expert_heads = nn.ModuleList([
            HyperExpertHead(feat_dim, feat_dim, out_channels * Q,
                            self.cond_dim) for _ in range(M)
        ])

        grid = torch.linspace(0, 1, Q).view(1, 1, Q, 1, 1)
        self.mu_grid = nn.Parameter(grid.repeat(1, out_channels, 1, 1, 1))
        self.w_init = nn.Parameter(0.1 * torch.randn(1, out_channels, Q, 1, 1))
        self.sigma_init = nn.Parameter(torch.ones(M, out_channels) * 0.1)
        self.chi_net = DeepGatedChi(out_channels * Q, out_channels)

        # v17: Cross-Channel Spectral Calibration
        self.spectral_calibrator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * Q, max(1, (out_channels * Q) // 4), 1),
            GELU(),
            nn.Conv2d(max(1, (out_channels * Q) // 4), out_channels * Q, 1),
            nn.Sigmoid())

    def forward(self, x, feat, illu_fea, illu_map):
        B, _, H, W = x.shape
        xi_basis = self.xi_net(x, feat).view(B, self.out_channels, self.Q, H,
                                             W)
        xi = x.unsqueeze(2) + torch.tanh(xi_basis) / self.Q

        cond_all = self.orchestrator(x)
        sigma_boost = torch.clamp(
            1.0 / (illu_map.mean(dim=1, keepdim=True) + 1e-4), 1.0,
            2.5).unsqueeze(2)
        psi_total = torch.zeros_like(xi)

        for i in range(self.M):
            cond = cond_all[:, self.cond_dim * i:self.cond_dim * (i + 1)]
            p_e = self.expert_heads[i](feat, illu_fea,
                                       cond).view(B, self.out_channels, self.Q,
                                                  3, H, W)
            w = self.w_init + p_e[:, :, :, 0]
            mu = self.mu_grid + torch.tanh(p_e[:, :, :, 1]) / self.Q
            s_base = self.sigma_init[i].view(1, self.out_channels, 1, 1, 1)
            sigma = (F.softplus(s_base + p_e[:, :, :, 2]) + 1e-6) * sigma_boost
            psi_total = psi_total + w * torch.exp(-0.5 * torch.pow(
                (xi - mu) / sigma, 2))

        psi_flat = psi_total.view(B, self.out_channels * self.Q, H, W)
        psi_flat = psi_flat * self.spectral_calibrator(
            psi_flat)  # v17 calibration
        out, _ = self.chi_net(psi_flat)
        return out, psi_flat


# --- Final HGSA-v17 Wrapper ---


class HGSA_v17(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, Q=7, M=5):
        super().__init__()
        HIDDEN_FEAT = 48
        self.encoder = Encoder2D(in_channels, HIDDEN_FEAT)
        self.vector_expert = HGSABlock(in_channels, out_channels, Q, M,
                                       HIDDEN_FEAT)
        self.chi_fusion = LaplacianGatedFusion(in_channels, out_channels)
        self.aux_proj = nn.Conv2d(Q * out_channels, out_channels, 1)

    def forward(self, x):
        feat, illu_fea, illu_map = self.encoder(x)
        usgs_out, sagf_out_raw = self.vector_expert(x, feat, illu_fea,
                                                    illu_map)

        final_out = self.chi_fusion(x, usgs_out, illu_map)

        if self.training:
            sagf_out = self.aux_proj(sagf_out_raw)
            return final_out, x + sagf_out

        return final_out
