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

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.laplacian_kernel = nn.Conv2d(in_channels,
                                          in_channels,
                                          3,
                                          padding=1,
                                          groups=in_channels)

        # Изменяем вход с (in+out) на (in+out+in), т.к. добавляем illu_map (3 канала)
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels + out_channels + in_channels,
                      32,
                      3,
                      padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, 1))
        self.refine = Advanced_GFFN(in_channels + out_channels, out_channels)
        self.temp = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.5)

    def forward(self, x_orig, usgs_out, illu_map):
        x_detail = x_orig - self.laplacian_kernel(x_orig)

        # Calculate the illumination-aware gate
        # We cat the manifold output with the original to find discrepancies
        #gate_input = torch.cat([x_orig, usgs_out], dim=1)
        #logits = self.gate_net(gate_input)
        
        # Конкатенируем оригинал, выход экспертов и карту освещенности
        gate_input = torch.cat([x_orig, usgs_out, illu_map], dim=1)
        raw_logits = self.gate_net(gate_input)

        # Illumination-based bias (as you had it, very effective for RYYB)
        #illu_bias = torch.pow(illu_map + 1e-6, 0.5) * self.temp
        #effective_gate = torch.sigmoid(logits - illu_bias)

        # Используем мягкое смещение через Sigmoid (Option A из нашего анализа)
        effective_gate = torch.sigmoid(raw_logits)

        blended = usgs_out * effective_gate + (x_orig + x_detail) * (
            1.0 - effective_gate)

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


# --- HGSABlock v17: Q-Dimensional Manifold ---


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


class LightMSAB(nn.Module):

    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(
                t, 'b (head c) h w -> b head c (h w)', head=self.num_heads),
            (q, k, v))

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out,
                        'b head c (h w) -> b (head c) h w',
                        head=self.num_heads,
                        h=h,
                        w=w)
        return self.project_out(out)


class HyperExpertHead(nn.Module):

    def __init__(self, feat_dim, hidden_dim, out_channels, dim_params,
                 cond_dim):
        super().__init__()
        # out_channels = 3, dim_params = 3 * Q
        all_channels = out_channels * dim_params
        # Spectral Reasoning Layer (MST++ Style)
        # This allows the expert to "see" the RYYB relationships globally
        self.spectral_reasoning = LightMSAB(feat_dim + 16 + cond_dim)
        # Parameter Projection
        self.input_proj = nn.Conv2d(feat_dim + 16 + cond_dim, hidden_dim, 1)
        # Use groups=out_channels (3) to give each channel its own dedicated
        # parameter weights while still living in the same head.
        self.main_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, all_channels, 1, groups=out_channels),
        )
        self.shortcut = nn.Conv2d(feat_dim + 16 + cond_dim, all_channels, 1)
        self.gamma = nn.Parameter(torch.ones(1, all_channels, 1, 1) * 0.1)

    def forward(self, feat, illu_fea, v):
        v = v.expand(-1, -1, feat.size(2), feat.size(3))
        combined = torch.cat([feat, illu_fea, v], dim=1)

        # Apply Spectral Attention
        # We use a residual connection here to keep the gradient path short
        combined = combined + self.spectral_reasoning(combined)

        # Project to manifold parameters (w, mu, sigma)
        x = self.input_proj(combined)
        return self.shortcut(combined) + self.main_branch(x) * self.gamma


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


class RecursiveFractalChi(nn.Module):

    def __init__(self, x_dim, psi_dim, out_dim):
        super().__init__()
        self.x_norm = LayerNorm(x_dim)
        # The working dimension is the manifold + the anchor
        combined_dim = psi_dim + x_dim

        # Level 1: Primary features (using expanded dim)
        self.dw1 = nn.Conv2d(combined_dim,
                             combined_dim,
                             3,
                             padding=1,
                             groups=combined_dim)

        # Level 2: High-frequency detail (must also use combined_dim)
        self.dw2 = nn.Conv2d(combined_dim,
                             combined_dim,
                             5,
                             padding=2,
                             groups=combined_dim)

        self.gate1 = nn.Sequential(nn.Conv2d(combined_dim, combined_dim, 1),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Conv2d(combined_dim, combined_dim, 1),
                                   nn.Sigmoid())

        self.proj_out = nn.Conv2d(combined_dim, out_dim, 1)

    def forward(self, psi, x):
        x = self.x_norm(x)
        # Concatenate anchor: [B, psi_dim + 3, H, W]
        psi_combined = torch.cat([psi, x], dim=1)

        # Fractal Gating logic on the combined manifold
        feat_h = self.dw2(psi_combined) * self.gate2(psi_combined)
        feat_m = self.dw1(psi_combined + feat_h) * self.gate1(feat_h)

        return self.proj_out(feat_m), self.gate1(feat_h)


class HGSABlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, Q=7, M=3, feat_dim=48):
        super().__init__()
        self.cond_dim = 8
        self.M, self.Q, self.out_channels = M, Q, out_channels
        self.xi_net = BasisAttention(in_channels, feat_dim, M, Q)
        self.orchestrator = SpectralOrchestrator(in_dim=in_channels,
                                                 cond_dim=M * self.cond_dim)

        # Head predicts 3 * Q parameters per channel
        self.expert_heads = nn.ModuleList([
            HyperExpertHead(feat_dim, feat_dim, out_channels, (3 * Q),
                            self.cond_dim) for _ in range(M)
        ])

        # Centripetal Initialization
        self.mu_init = nn.Parameter(torch.ones(1, out_channels, Q, 1, 1) * 0.5)
        self.mu_scale = nn.Parameter(torch.ones(1, out_channels, Q, 1, 1) * 4.5)
        self.w_init = nn.Parameter(0.1 * torch.randn(1, out_channels, Q, 1, 1))
        self.sigma_init = nn.Parameter(torch.ones(M, out_channels, Q) * 0.2)

        self.spectral_calibrator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * Q, max(1, (out_channels * Q) // 4), 1),
            GELU(),
            nn.Conv2d(max(1, (out_channels * Q) // 4), out_channels * Q, 1),
            nn.Sigmoid())

        self.chi_net = RecursiveFractalChi(in_channels, out_channels * Q,
                                           out_channels)

    def forward(self, x, feat, illu_fea, illu_map):
        B, C, H, W = x.shape
        xi_basis = self.xi_net(x, feat).view(B, 1, self.Q, H,
                                             W).expand(-1, self.out_channels,
                                                       -1, -1, -1)
        xi = x.unsqueeze(2) + torch.tanh(xi_basis)

        cond_all = self.orchestrator(x)
        sigma_boost = torch.clamp(
            1.0 / (illu_map.mean(dim=1, keepdim=True) + 1e-4), 1.0,
            2.5).unsqueeze(2)
        psi_total = torch.zeros(B,
                                self.out_channels,
                                self.Q,
                                H,
                                W,
                                device=x.device)

        for i in range(self.M):
            cond = cond_all[:, self.cond_dim * i:self.cond_dim * (i + 1)]
            # p_e shape: [B, C, 3*Q, H, W]
            p_e = self.expert_heads[i](feat, illu_fea,
                                       cond).view(B, self.out_channels, 3,
                                                  self.Q, H, W)

            w = self.w_init + p_e[:, :, 0] # [-1,2]
            mu_scale = F.softplus(self.mu_scale)
            mu = torch.tanh(self.mu_init + p_e[:, :, 1]) * mu_scale  + 0.5
            s_base = self.sigma_init[i].view(1, self.out_channels, self.Q, 1,
                                             1)
            sigma = (F.softplus(s_base + p_e[:, :, 2]) + 0.01) * sigma_boost

            gaussian_kernel = torch.exp(-0.5 * torch.pow((xi - mu) / sigma, 2))
            psi_total = psi_total + w * gaussian_kernel

        psi_flat = psi_total.view(B, self.out_channels * self.Q, H, W)
        psi_flat = psi_flat * self.spectral_calibrator(psi_flat)
        out, _ = self.chi_net(psi_flat, x)
        return out, psi_flat


class HGSA_v17(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, Q=7, M=3):
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
