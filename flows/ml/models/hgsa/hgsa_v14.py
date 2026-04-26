import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --- Базовые компоненты и Активации ---


class GELU(nn.Module):

    def forward(self, x):
        return F.gelu(x)


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dim() == 4:
            h, w = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.body(x)
            return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.body(x)


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
            nn.Conv2d(dim, dim // 2, kernel_size=1))
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illu_feat):
        b, c, h, w = x.shape
        q, k, a = self.q_proj(x), self.k_proj(x), self.a_proj(x)
        v = self.v_proj(x) * illu_feat
        q, k, a, v = [
            rearrange(t,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads) for t in (q, k, a, v)
        ]
        q, k, a = [F.normalize(t, dim=-1) for t in (q, k, a)]
        attn_a = (q @ a.transpose(-2, -1)) * self.temperature_a
        attn_k = (a @ k.transpose(-2, -1)) * self.temperature_v
        out = rearrange(attn_a.softmax(dim=-1) @ (attn_k.softmax(dim=-1) @ v),
                        'b head c (h w) -> b (head c) h w',
                        head=self.num_heads,
                        h=h,
                        w=w)
        return self.project_out(out)


# --- MST++ / MSAB Компоненты ---


class MaskGuidedMechanism(nn.Module):

    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat,
                                    n_feat,
                                    kernel_size=5,
                                    padding=2,
                                    bias=True,
                                    groups=n_feat)

    def forward(self, mask):
        mask = self.conv1(mask)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask)))
        return mask * attn_map + mask


class MS_MSA(nn.Module):

    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.mm = MaskGuidedMechanism(dim)

    def forward(self, x_in, mask=None):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q, k, v = map(
            lambda t: rearrange(t(x), 'b n (h d) -> b h n d', h=self.num_heads
                                ), (self.to_q, self.to_k, self.to_v))

        mask_attn = self.mm(mask.permute(0, 3, 1,
                                         2)).permute(0, 2, 3,
                                                     1).reshape(b, h * w, c)
        mask_attn = rearrange(mask_attn,
                              'b n (h d) -> b h n d',
                              h=self.num_heads)
        v = v * mask_attn

        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).permute(0, 3, 1,
                               2).reshape(b, h * w,
                                          self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(rearrange(x_in, 'b h w c -> b c h w')).permute(
            0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult,
                      dim * mult,
                      3,
                      1,
                      1,
                      bias=False,
                      groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):

    def __init__(self, dim, dim_head, heads, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList([
                    MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                    nn.LayerNorm(dim),
                    FeedForward(dim=dim)
                ]))

    def forward(self, x, mask):
        x, m = x.permute(0, 2, 3, 1), mask.permute(0, 2, 3, 1)
        for (attn, norm, ff) in self.blocks:
            x = attn(x, mask=m) + x
            x = ff(norm(x)) + x
        return x.permute(0, 3, 1, 2)


# --- Advanced GFFN ---


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
        self.spectral_calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_dim * 2, out_dim * 2, 1),
            nn.Sigmoid())
        self.project_out = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, x):
        combined = self.project_in(x)
        combined = combined * self.spectral_calibration(combined)
        x1, x2 = combined.chunk(2, dim=1)
        return self.project_out(
            self.dwconv_3x3(x1) * torch.sigmoid(self.dwconv_5x5(x2)))


# --- Encoder2D v13 ---


class RGB_IlluminationEstimator(nn.Module):

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1)
        self.depth_conv = nn.Conv2d(n_fea_middle,
                                    n_fea_middle,
                                    kernel_size=5,
                                    padding=2,
                                    groups=n_fea_middle)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1)

    def forward(self, img):
        input_feat = torch.cat([img, img.mean(dim=1, keepdim=True)], dim=1)
        illu_fea = self.depth_conv(self.conv1(input_feat))
        illu_map = torch.exp(torch.clamp(self.conv2(illu_fea), -2, 2))
        return illu_fea, illu_map


class SpectralTransformerBlock(nn.Module):

    def __init__(self, in_channel, num_heads, bias):
        super().__init__()
        self.norm1 = LayerNorm(in_channel)
        self.attn = Attention(in_channel, num_heads, bias)
        self.norm2 = LayerNorm(in_channel)
        self.ffn = Advanced_GFFN(in_channel, in_channel)

    def forward(self, x, illu_feat):
        x = x + self.attn(self.norm1(x), illu_feat)
        x = x + self.ffn(self.norm2(x))
        return x


class Encoder2D_v13(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.estimator = RGB_IlluminationEstimator(16, in_channels + 1,
                                                   in_channels)
        self.down1 = nn.PixelUnshuffle(2)
        self.trans1 = SpectralTransformerBlock(12, 3, True)
        self.illu_down1 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(16, 12, 1))
        self.down2 = nn.PixelUnshuffle(2)
        self.trans2 = SpectralTransformerBlock(48, 3, True)
        self.illu_down2 = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(16, 48, 1))
        self.up1 = nn.Upsample(scale_factor=2,
                               mode='bilinear',
                               align_corners=False)
        self.up2 = nn.Upsample(scale_factor=4,
                               mode='bilinear',
                               align_corners=False)
        self.conv_out = nn.Sequential(
            LayerNorm(in_channels + 12 + 48),
            Advanced_GFFN(in_channels + 12 + 48, out_channels))

    def forward(self, x):
        illu_fea, illu_map = self.estimator(x)
        x_orig = x * illu_map
        x1 = self.trans1(self.down1(x_orig), self.illu_down1(illu_fea))
        x2 = self.trans2(self.down2(x1), self.illu_down2(illu_fea))
        out = torch.cat([x_orig, self.up1(x1), self.up2(x2)], dim=1)
        return self.conv_out(out)


# --- НОВЫЕ КОМПОНЕНТЫ V14 ---


class Orchestrator_v14(nn.Module):

    def __init__(self, in_channels, encoder_channels, hidden_channels,
                 out_channels, num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.out_channels = out_channels
        self.in_proj = nn.Conv2d(in_channels + encoder_channels,
                                 hidden_channels, 1)
        self.msab = MSAB(dim=hidden_channels,
                         dim_head=hidden_channels // 2,
                         heads=2,
                         num_blocks=2)
        self.proj = nn.Conv2d(hidden_channels,
                              2 * num_gaussians * out_channels, 1)

    def forward(self, x, encoder_feat):
        B, _, H, W = x.shape
        feat = self.in_proj(torch.cat([x, encoder_feat], dim=1))
        feat = self.msab(feat, feat)
        data = self.proj(feat)
        raw_gates = data[:, :self.num_gaussians * self.out_channels].view(
            B, self.num_gaussians, self.out_channels, H, W)
        gates = F.softmax(raw_gates, dim=1)
        taus = data[:, self.num_gaussians * self.out_channels:].view(
            B, self.num_gaussians, self.out_channels, H, W)
        return gates, taus


class HyperExpertHead_v14(nn.Module):

    def __init__(self, feat_dim, xi_dim, hidden_dim, out_channels):
        super().__init__()
        self.input_proj = nn.Conv2d(feat_dim + xi_dim, hidden_dim, 1)
        self.main_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.SiLU(), nn.Conv2d(hidden_dim, 3 * out_channels, 1))
        self.shortcut = nn.Conv2d(feat_dim + xi_dim, 3 * out_channels, 1)
        self.gamma = nn.Parameter(torch.ones(1, 3 * out_channels, 1, 1) * 0.1)

    def forward(self, xi, feat):
        combined = torch.cat([xi, feat], dim=1)
        h = self.input_proj(combined)
        return self.shortcut(combined) + self.main_branch(h) * self.gamma


class ChiNet_v14(nn.Module):

    def __init__(self, in_channels, feat_channels, encoder_channels,
                 hidden_channels, out_channels):
        super().__init__()
        self.pre = nn.Conv2d(in_channels + feat_channels + encoder_channels,
                             hidden_channels, 1)
        self.msab = MSAB(dim=hidden_channels,
                         dim_head=hidden_channels // 4,
                         heads=4,
                         num_blocks=3)
        self.post = nn.Conv2d(hidden_channels + in_channels, out_channels, 1)

    def forward(self, x, psi_total, encoder_feat):
        combined = torch.cat([x, psi_total, encoder_feat], dim=1)
        feat = self.msab(self.pre(combined), self.pre(combined))
        return self.post(torch.cat([x, feat], dim=1))


# --- HGSA v14 Smart Orchestra ---


class HGSA_v14(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_gaussians=9,
                 g_channels=5):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.g_channels = g_channels
        HIDDEN_FEAT = 32

        self.encoder = Encoder2D_v13(in_channels, HIDDEN_FEAT)
        self.xi_net = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1),
                                    nn.SiLU(), nn.Conv2d(16, g_channels, 1))
        self.orchestrator = Orchestrator_v14(in_channels, HIDDEN_FEAT,
                                             HIDDEN_FEAT, g_channels,
                                             num_gaussians)
        self.expert_heads = nn.ModuleList([
            HyperExpertHead_v14(HIDDEN_FEAT, g_channels, HIDDEN_FEAT,
                                g_channels) for _ in range(num_gaussians)
        ])

        self.mu_offsets = nn.Parameter(
            torch.linspace(0.01, 0.99, num_gaussians))
        self.w_init = nn.Parameter(torch.randn(num_gaussians) * 0.1)
        self.sigma_init = nn.Parameter(torch.ones(num_gaussians) * 0.05)

        self.chi_net = ChiNet_v14(in_channels, g_channels, HIDDEN_FEAT,
                                  HIDDEN_FEAT, out_channels)
        self.usgs_to_img = nn.Conv2d(g_channels, out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.encoder(x)
        xi = self.xi_net(x)
        gates, taus = self.orchestrator(x, feat)

        psi_total = torch.zeros_like(xi)
        p_list = []

        for i in range(self.num_gaussians):
            gate, tau = gates[:, i], taus[:, i]
            p_e = self.expert_heads[i](xi, feat).view(B, -1, 3, H, W)

            # --- UNBOUND APPROXIMATION ---
            w = self.w_init[i] + p_e[:, :, 0]
            mu = self.mu_offsets[i] + p_e[:, :, 1]
            sigma = F.softplus(self.sigma_init[i] + p_e[:, :, 2] + tau) + 1e-6

            psi_i = gate * w * torch.exp(-0.5 * torch.pow(
                (xi - mu) / sigma, 2))
            psi_total = psi_total + psi_i

            if self.training:
                p_list.append(torch.stack([w, mu, sigma, gate, tau], dim=2))

        main_out = self.chi_net(x, psi_total, feat)
        usgs_out = x + self.usgs_to_img(psi_total)

        return (main_out, usgs_out, p_list) if self.training else main_out
