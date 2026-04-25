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


# --- MST++ / MSAB Оригинальные компоненты ---


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

        # 1. Получаем маску аттеншена в пространстве [B, H, W, C]
        mask_attn = self.mm(mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 2. Выпрямляем маску в [B, N, C]
        mask_attn = mask_attn.reshape(b, h * w, c)

        # 3. Трансформируем q, k, v в головы: [b head n d]
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp))

        # 4. ВАЖНО: Трансформируем маску в ту же структуру голов [b head n d]
        mask_attn = rearrange(mask_attn,
                              'b n (h d) -> b h n d',
                              h=self.num_heads)

        # 5. Теперь умножение v * mask_attn корректно
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

        # Позиционное кодирование (pos_emb ожидает [B, C, H, W])
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


# --- Advanced GFFN (Multi-Scale + Spectral Calibration) ---


class Advanced_GFFN(nn.Module):

    def __init__(self, in_dim, out_dim):  # Явно разделяем вход и выход
        super().__init__()
        self.project_in = nn.Conv2d(in_dim, out_dim * 2,
                                    1)  # Проецируем в 2 * out_dim

        internal_dim = out_dim
        self.dwconv_3x3 = nn.Conv2d(internal_dim,
                                    internal_dim,
                                    3,
                                    padding=1,
                                    groups=internal_dim)
        self.dwconv_5x5 = nn.Conv2d(internal_dim,
                                    internal_dim,
                                    5,
                                    padding=2,
                                    groups=internal_dim)

        self.spectral_calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(internal_dim * 2, internal_dim * 2, 1), nn.Sigmoid())
        self.project_out = nn.Conv2d(internal_dim, out_dim, 1)

    def forward(self, x):
        combined = self.project_in(x)
        combined = combined * self.spectral_calibration(combined)

        x1, x2 = combined.chunk(2, dim=1)
        x1 = self.dwconv_3x3(x1)
        x2 = self.dwconv_5x5(x2)

        return self.project_out(x1 * torch.sigmoid(x2))


# --- Модифицированный Encoder2D компоненты ---


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
        mean_c = img.mean(dim=1, keepdim=True)
        input_feat = torch.cat([img, mean_c], dim=1)
        illu_fea = self.depth_conv(self.conv1(input_feat))
        illu_map = self.conv2(illu_fea)
        # Экспоненциальное усиление для расширения динамического диапазона (v13)
        return illu_fea, torch.exp(torch.clamp(illu_map, -2, 2))


class SpectralTransformerBlock(nn.Module):

    def __init__(self, in_channel, num_heads, bias):
        super().__init__()
        self.norm1 = LayerNorm(in_channel)
        # Используем стандартный Attention из v11, так как он эффективен
        self.attn = Attention(in_channel, num_heads, bias)
        self.norm2 = LayerNorm(in_channel)
        # Здесь вход и выход равны in_channel
        self.ffn = Advanced_GFFN(in_channel, in_channel)

    def forward(self, x, illu_feat):
        x = x + self.attn(self.norm1(x), illu_feat)
        x = x + self.ffn(self.norm2(x))
        return x


class Encoder2D_v13(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.estimator = RGB_IlluminationEstimator(16, in_dim + 1, in_dim)
        self.down1 = nn.PixelUnshuffle(2)
        self.trans1 = SpectralTransformerBlock(12, 3,
                                               True)  # 3*4=12 после unshuffle
        self.illu_down1 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(16, 12, 1))

        self.down2 = nn.PixelUnshuffle(2)
        self.trans2 = SpectralTransformerBlock(48, 3, True)  # 12*4=48
        self.illu_down2 = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(16, 48, 1))

        self.up1 = nn.Upsample(scale_factor=2,
                               mode='bilinear',
                               align_corners=False)
        self.up2 = nn.Upsample(scale_factor=4,
                               mode='bilinear',
                               align_corners=False)

        # Вход в conv_out: in_dim(3) + 12 + 48 = 63
        self.conv_out = nn.Sequential(
            LayerNorm(in_dim + 12 + 48),
            Advanced_GFFN(in_dim + 12 + 48,
                          out_dim)  # Здесь должно быть 63 -> 32
        )

    def forward(self, x):
        illu_fea, illu_map = self.estimator(x)
        x_orig = x * illu_map

        x1 = self.trans1(self.down1(x_orig), self.illu_down1(illu_fea))
        x2 = self.trans2(self.down2(x1), self.illu_down2(illu_fea))

        out = torch.cat([x_orig, self.up1(x1), self.up2(x2)], dim=1)
        return self.conv_out(out)


# --- Orchestrator (Contrast Gated) ---


class Orchestrator_v13(nn.Module):
    def __init__(self, in_channels, hidden_feat, num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.in_channels = in_channels
        
        # Проекция входного изображения в скрытое пространство
        self.in_proj = nn.Conv2d(in_channels, hidden_feat, 1)
        
        # Пространственно-спектральный анализ на основе MSAB
        self.msab = MSAB(
            dim=hidden_feat,
            dim_head=hidden_feat // 2,
            heads=2,
            num_blocks=2,
        )
        
        # Финальная проекция в параметры ворот (gates) и смещений (taus)
        # Выход: num_gaussians * in_channels (для gates) + num_gaussians * in_channels (для taus)
        self.proj = nn.Conv2d(
            hidden_feat,
            2 * num_gaussians * in_channels,
            1,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Работаем напрямую с входным изображением x
        feat = self.in_proj(x)
        feat = self.msab(feat, feat)
        data = self.proj(feat)
        
        # Разделяем выход на ворота и смещения
        # Gates (0 : num_gaussians * C)
        raw_gates = data[:, :self.num_gaussians * C].view(
            B, self.num_gaussians, C, H, W)
        gates = F.softmax(raw_gates, dim=1)
        
        # Taus (num_gaussians * C : end)
        taus = data[:, self.num_gaussians * C:].view(
            B, self.num_gaussians, C, H, W)
        
        return gates, taus


# --- Hyper-FFN Heads (Contrast Gated) ---


class HyperFFN_Head(nn.Module):

    def __init__(self, in_feat, hidden_feat, out_channels):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_feat, hidden_feat, 1), nn.SiLU(),
            nn.Conv2d(hidden_feat, 3 * out_channels, 1))
        self.contrast_gate = nn.Sequential(nn.Conv2d(in_feat, 1, 3, padding=1),
                                           nn.Sigmoid())

    def forward(self, feat):
        params = self.main_path(feat)
        gate = self.contrast_gate(feat)
        return params * gate


# --- HGSA v13 Smart Orchestra ---


class HGSA_v13(nn.Module): # Ваша текущая версия

    def __init__(self, in_channels=3, out_channels=3, num_gaussians=7):
        super().__init__()
        self.num_gaussians = num_gaussians
        HIDDEN_FEAT = 32

        self.encoder = Encoder2D_v13(in_channels, out_dim=HIDDEN_FEAT)
        
        # Инициализация нового оркестратора
        self.orchestrator = Orchestrator_v13(
            in_channels=in_channels, 
            hidden_feat=HIDDEN_FEAT, 
            num_gaussians=num_gaussians
        )

        self.xi_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, in_channels, 1),
        )

        self.expert_heads = nn.ModuleList([
            HyperFFN_Head(HIDDEN_FEAT, HIDDEN_FEAT, in_channels)
            for _ in range(num_gaussians)
        ])

        # Gated Chi-Net (v13)
        self.chi_pre = nn.Conv2d(in_channels * 3, HIDDEN_FEAT, 1)
        self.chi_gate = nn.Sequential(
            nn.Conv2d(HIDDEN_FEAT, HIDDEN_FEAT, 1),
            nn.Sigmoid(),
        )
        self.chi_msab = MSAB(
            dim=HIDDEN_FEAT,
            dim_head=HIDDEN_FEAT // 4,
            heads=4,
            num_blocks=3,
        )
        self.chi_post = nn.Conv2d(HIDDEN_FEAT, out_channels, 1)

        self.x_norm = LayerNorm(in_channels)
        self.usgs_to_img = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        xi = self.xi_net(x)
        feat = self.encoder(x)

        # Вызов оркестратора: получаем маски и центры напрямую из x
        gates_normed, taus_all = self.orchestrator(x)

        psi_total = torch.zeros_like(xi)
        p_list = []

        for i in range(self.num_gaussians):
            gate = gates_normed[:, i]
            tau = taus_all[:, i]

            p_e = self.expert_heads[i](feat).view(B, C, 3, H, W)
            
            # w - positive and negative to approximate residual x - psi_total
            # mu - must be close to xi, approximate only in neighborhood of xi
            # sigma - responsible for surface shape, gets more freedom
            w = p_e[:, :, 0, :, :]
            mu = torch.tanh(p_e[:, :, 1, :, :]) * 1.0
            sigma = torch.sigmoid(p_e[:, :, 2, :, :] + tau) * 4.0 + 0.02

            diff = (xi - mu) / (sigma + 1e-6)
            psi_i = gate * w * torch.exp(-0.5 * diff**2)
            psi_total = psi_total + psi_i
            
            if self.training:
                p_list.append(torch.stack([w, mu, sigma, gate, tau], dim=2))

        # Финальная сборка (Chi-Net)
        usgs_out = self.usgs_to_img(psi_total) + x
        combined = torch.cat([x, self.x_norm(x), psi_total], dim=1)

        chi_feat = self.chi_pre(combined)
        chi_feat = chi_feat * self.chi_gate(chi_feat)
        chi_feat = self.chi_msab(chi_feat, chi_feat)
        main_out = self.chi_post(chi_feat)

        return (main_out, usgs_out, p_list) if self.training else main_out
