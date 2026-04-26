"""
Hybrid Parametric-Attention USGS объединяет:
- Глобальный контекст (Encoder)
- Обучаемый словарь базисов (Basis Attention)
- Физику Гауссовых полей (HGSA)
- Топологию суперпозиции (USGS)
"""
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


class Encoder2D(nn.Module):

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


class Orchestrator(nn.Module):

    def __init__(
        self,
        in_channels,
        encoder_channels,
        hidden_channels,
        out_channels,
        num_gaussians,
    ):
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


class HyperExpertHead(nn.Module):

    def __init__(
        self,
        feat_dim,
        xi_dim,
        hidden_dim,
        out_channels,
    ):
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


class ChiNet(nn.Module):

    def __init__(
        self,
        in_channels,
        feat_channels,
        encoder_channels,
        hidden_channels,
        out_channels,
    ):
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


# --- Dynamic Query-based Basis ---


class BasisAttention(nn.Module):

    def __init__(self, feat_dim, Q):
        super().__init__()
        self.Q = Q
        # Обучаемый словарь базисных состояний (набор "цветовых концептов")
        # M = количество концептов (например, 16)
        self.num_concepts = 16
        self.basis_v = nn.Parameter(torch.randn(self.num_concepts, Q))

        self.q_proj = nn.Conv2d(feat_dim, self.num_concepts, 1)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, feat):
        B, C, H, W = feat.shape
        # Генерируем Query из признаков энкодера
        # q: [B, num_concepts, H, W]
        q = self.q_proj(feat)

        # Вычисляем веса внимания (насколько каждый пиксель похож на базисный концепт)
        # attn: [B, num_concepts, H*W]
        attn = F.softmax(q.view(B, self.num_concepts, -1) * self.temperature,
                         dim=1)

        # Извлекаем значения из обучаемого словаря
        # res: [B, Q, H*W] = [B, Q, num_concepts] @ [B, num_concepts, H*W]
        v = self.basis_v.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
        res = torch.bmm(v, attn)

        return res.view(B, self.Q, H, W)


# --- HGSA v14 Smart Orchestra ---


class HGSABlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, Q=7, M=5, feat_dim=32):
        super().__init__()
        self.M = M
        self.Q = Q
        self.feat_dim = feat_dim

        self.xi_net = BasisAttention(feat_dim, Q)

        self.orchestrator = Orchestrator(
            in_channels=in_channels,
            out_channels=Q,
            encoder_channels=feat_dim,
            hidden_channels=feat_dim,
            num_gaussians=M,
        )

        self.expert_heads = nn.ModuleList([
            HyperExpertHead(
                feat_dim=feat_dim,
                xi_dim=Q,
                hidden_dim=feat_dim,
                out_channels=Q,
            ) for _ in range(M)
        ])

        self.mu_offsets = nn.Parameter(torch.linspace(0.01, 0.99, M))
        self.w_init = nn.Parameter(torch.randn(M) * 0.1)
        self.sigma_init = nn.Parameter(torch.ones(M) * 0.05)

        self.chi_net = ChiNet(
            in_channels=in_channels,
            feat_channels=Q,
            encoder_channels=feat_dim,
            hidden_channels=feat_dim,
            out_channels=out_channels,
        )

        self.sagf_to_img = nn.Conv2d(Q, out_channels, 1)

    def forward(self, x, feat):
        B, C, H, W = x.shape

        xi = self.xi_net(feat)
        gates, taus = self.orchestrator(x, feat)

        psi_total = torch.zeros_like(xi)

        for i in range(self.M):
            gate, tau = gates[:, i], taus[:, i]
            p_e = self.expert_heads[i](xi, feat).view(B, -1, 3, H, W)

            # --- UNBOUND APPROXIMATION ---
            w = self.w_init[i] + p_e[:, :, 0]
            mu = self.mu_offsets[i] + p_e[:, :, 1]
            sigma = F.softplus(self.sigma_init[i] + p_e[:, :, 2] + tau) + 1e-6

            psi_i = gate * w * torch.exp(-0.5 * torch.pow(
                (xi - mu) / sigma, 2))
            psi_total = psi_total + psi_i

        usgs_out = self.chi_net(x, psi_total, feat)

        if self.training:
            return usgs_out, psi_total

        return usgs_out


class HGSA_v15(nn.Module):
    """
    Финальная архитектура. 
    Содержит общий Encoder и ModuleList из HGSA блоков по числу каналов.
    """

    def __init__(self, in_channels=3, out_channels=3, Q=7, M=5):
        super().__init__()
        HIDDEN_FEAT = 32

        # Общий энкодер признаков (физика сцены)
        self.encoder = Encoder2D(in_channels, HIDDEN_FEAT)

        # Ансамбль экспертов по количеству выходных каналов
        self.channel_experts = nn.ModuleList([
            HGSABlock(in_channels=in_channels,
                      out_channels=1,
                      Q=Q,
                      M=M,
                      feat_dim=HIDDEN_FEAT) for _ in range(out_channels)
        ])

        # Вспомогательная ветка для стабилизации (aux loss)
        self.aux_proj = nn.Conv2d(Q * out_channels, out_channels, 1)

    def forward(self, x):
        feat = self.encoder(x)

        ch_usgs_list = []
        ch_sagf_list = []

        # Каждый блок HGSA_v14 обрабатывает свой канал независимо
        for _, expert in enumerate(self.channel_experts):
            if self.training:
                ch_usgs, ch_sagf = expert(x, feat)
                ch_sagf_list.append(ch_sagf)
            else:
                ch_usgs = expert(x, feat)
            ch_usgs_list.append(ch_usgs)

        # Склеиваем результат в RGB
        usgs_out = torch.cat(ch_usgs_list, dim=1)

        if self.training:
            # Для aux loss суммируем вклады всех полей
            sagf_out = torch.cat(ch_sagf_list, dim=1)
            sagf_out = x + self.aux_proj(sagf_out)
            return usgs_out, sagf_out

        return usgs_out
