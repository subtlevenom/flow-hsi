import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# --- MST++ / MSAB Оригинальные компоненты ---


class GELU(nn.Module):

    def forward(self, x):
        return F.gelu(x)


class MaskGuidedMechanism(nn.Module):

    def __init__(self, n_feat):
        super(MaskGuidedMechanism, self).__init__()
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
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        mask_attn = self.mm(mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        mask_attn = mask_attn.expand([b, h, w, c])

        q, k, v, mask_attn = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2)))
        v = v * mask_attn

        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2, eps=1e-6)
        k = F.normalize(k, dim=-1, p=2, eps=1e-6)

        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).permute(0, 3, 1,
                               2).reshape(b, h * w,
                                          self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w,
                                           c).permute(0, 3, 1,
                                                      2)).permute(0, 2, 3, 1)
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
        x = x.permute(0, 2, 3, 1)
        m = mask.permute(0, 2, 3, 1)
        for (attn, norm, ff) in self.blocks:
            x = attn(x, mask=m) + x
            x = ff(norm(x)) + x
        return x.permute(0, 3, 1, 2)


# --- Базовые утилиты ---
class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class DWTForward(nn.Module):

    def forward(self, x):
        return F.pixel_unshuffle(x, 2)


# --- Компоненты Encoder2D ---
class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        self.out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features, hidden_features, 1)
        self.depthwise = nn.Conv2d(hidden_features,
                                   hidden_features,
                                   3,
                                   padding=1,
                                   groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features, self.out_features, 1)
        self.act_layer = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.pointwise2(
            self.act_layer(self.depthwise(self.pointwise1(x))))


class IlluminationEstimator(nn.Module):

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1)
        self.depth_conv = nn.Conv2d(n_fea_middle,
                                    n_fea_middle,
                                    kernel_size=5,
                                    padding=2,
                                    groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        illu_fea = self.depth_conv(self.conv1(input))
        illu_map = self.conv2(illu_fea)
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


class TransformerBlock(nn.Module):

    def __init__(self, in_channel, mid_channel, out_channel, num_heads, bias):
        super().__init__()
        self.norm1 = LayerNorm(in_channel)
        self.attn = Attention(in_channel, num_heads, bias)
        self.norm2 = LayerNorm(in_channel)
        self.ffn = FFN(in_channel, mid_channel, out_channel)

    def forward(self, x, illu_feat):
        x = x + self.attn(self.norm1(x), illu_feat)
        x = x + self.ffn(self.norm2(x))
        return x


class Encoder2D(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.estimator = IlluminationEstimator(12, in_dim + 1, in_dim)
        self.down1 = DWTForward()
        self.trans1 = TransformerBlock(12, 12, 12, 3, True)
        self.illu_down1 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(12, 12, 1))
        self.down2 = DWTForward()
        self.trans2 = TransformerBlock(48, 48, 48, 3, True)
        self.illu_down2 = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(12, 48, 1))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv_out = nn.Sequential(LayerNorm(in_dim + 12 + 48),
                                      FFN(in_dim + 12 + 48, out_dim, out_dim))

    def forward(self, x):
        illu_fea, illu_map = self.estimator(x)
        x_orig = x * illu_map + x
        x1 = self.trans1(self.down1(x_orig), self.illu_down1(illu_fea))
        x2 = self.trans2(self.down2(x1), self.illu_down2(illu_fea))
        out = torch.cat([x_orig, self.up1(x1), self.up2(x2)], dim=1)
        return self.conv_out(out)


# --- HGSA v11 Smart Orchestra ---
class HGSA_v11(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, num_gaussians=7):
        super().__init__()
        self.num_gaussians = num_gaussians
        HIDDEN_FEAT = 32

        self.encoder = Encoder2D(in_channels, out_dim=HIDDEN_FEAT)
        self.xi_net = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1),
                                    nn.SiLU(), nn.Conv2d(16, in_channels, 1),
                                    LayerNorm(in_channels))

        self.expert_heads = nn.ModuleList([
            FFN(HIDDEN_FEAT, HIDDEN_FEAT, 3 * in_channels)
            for _ in range(num_gaussians)
        ])

        # Orchestrator с оригинальным MSAB
        self.orchestrator_msab = MSAB(dim=HIDDEN_FEAT,
                                      dim_head=16,
                                      heads=2,
                                      num_blocks=2)
        self.orchestrator_proj = nn.Conv2d(HIDDEN_FEAT,
                                           2 * num_gaussians * in_channels, 1)

        # Chi-Net с оригинальным MSAB
        self.chi_pre = nn.Conv2d(in_channels * 3, HIDDEN_FEAT, 1)
        self.chi_msab = MSAB(dim=HIDDEN_FEAT,
                             dim_head=16,
                             heads=2,
                             num_blocks=2)
        self.chi_post = nn.Conv2d(HIDDEN_FEAT, out_channels, 1)

        self.x_norm = LayerNorm(in_channels)
        self.usgs_to_img = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        xi = self.xi_net(x)
        feat = self.encoder(x)

        # Orchestrator Logic
        orch_feat = self.orchestrator_msab(feat, feat)  # feat как маска
        orch_data = self.orchestrator_proj(orch_feat)

        raw_gates = orch_data[:, :self.num_gaussians * C].view(
            B, self.num_gaussians, C, H, W)
        gates_normed = F.softmax(raw_gates, dim=1)
        taus_all = torch.sigmoid(orch_data[:, self.num_gaussians * C:]).view(
            B, self.num_gaussians, C, H, W) * 2.0 + 0.2

        psi_total = torch.zeros_like(xi)
        p_list = []

        for i in range(self.num_gaussians):
            p_e = self.expert_heads[i](feat).view(B, C, 3, H, W)
            w = torch.tanh(p_e[:, :, 0, :, :]) * 2.0
            mu = torch.tanh(p_e[:, :, 1, :, :]) * 4.0
            sigma = torch.sigmoid(p_e[:, :, 2, :, :]) * 4.0 + 0.05

            gate, tau = gates_normed[:, i], taus_all[:, i]
            diff = (xi - mu) / (sigma * tau + 1e-6)
            psi_i = gate * w * torch.exp(-0.5 * diff**2)
            psi_total = psi_total + psi_i
            if self.training:
                p_list.append(torch.stack([w, mu, sigma, gate, tau], dim=2))

        # Chi-Net Logic (Конкатенация как в v10)
        usgs_out = self.usgs_to_img(psi_total) + x
        combined = torch.cat([x, self.x_norm(x), psi_total], dim=1)

        chi_feat = self.chi_pre(combined)
        chi_feat = self.chi_msab(chi_feat, chi_feat)
        main_out = self.chi_post(chi_feat)

        return (main_out, usgs_out, p_list) if self.training else main_out
