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

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.estimator = RGB_IlluminationEstimator(16, in_channels + 1,
                                                   in_channels)
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
            LayerNorm(in_channels + 12 + 48),
            Advanced_GFFN(in_channels + 12 + 48,
                          out_channels)  # Здесь должно быть 63 -> 32
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

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Проекция входного изображения в скрытое пространство
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, 1)

        # Пространственно-спектральный анализ на основе MSAB
        self.msab = MSAB(
            dim=hidden_channels,
            dim_head=hidden_channels // 2,
            heads=2,
            num_blocks=2,
        )

        # Финальная проекция в параметры ворот (gates) и смещений (taus)
        # Выход: num_gaussians * in_channels (для gates) + num_gaussians * in_channels (для taus)
        self.proj = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=2 * num_gaussians * out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Работаем напрямую с входным изображением x
        feat = self.in_proj(x)
        feat = self.msab(feat, feat)
        data = self.proj(feat)

        # Разделяем выход на ворота и смещения
        # Gates (0 : num_gaussians * C)
        raw_gates = data[:, :self.num_gaussians * self.out_channels].view(
            B, self.num_gaussians, self.out_channels, H, W)
        gates = F.sigmoid(raw_gates)
        # gates = F.softmax(raw_gates, dim=1)

        # Taus (num_gaussians * C : end)
        taus = data[:, self.num_gaussians * self.out_channels:].view(
            B, self.num_gaussians, self.out_channels, H, W)

        return gates, taus


# --- Chi-net (Aggregator) ---


class ChiNet_v13(nn.Module):

    def __init__(self, in_channels, feat_channels, hidden_channels,
                 out_channels):
        super().__init__()

        self.pre = nn.Conv2d(feat_channels + in_channels, hidden_channels, 1)

        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.Sigmoid(),
        )

        self.msab = MSAB(
            dim=hidden_channels,
            dim_head=hidden_channels // 4,
            heads=4,
            num_blocks=3,
        )

        # Финальный слой: принимает признаки коррекции + исходный резкий x
        self.post = nn.Conv2d(hidden_channels + in_channels, out_channels, 1)

    def forward(self, x, psi_total):
        # 1. Сборка входного признака
        combined = torch.cat([x, psi_total], dim=1)

        # 2. Обработка дельты
        feat = self.pre(combined)
        feat = feat * self.gate(feat)
        feat = self.msab(feat, feat)

        # 3. Накладываем выученную коррекцию на чистый исходный x
        out = self.post(torch.cat([x, feat], dim=1))
        return out


# --- Hyper-FFN Heads (Contrast Gated) ---


class HyperFFN_Head(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1), nn.SiLU(),
            nn.Conv2d(hidden_channels, 3 * out_channels, 1))
        self.contrast_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, feat):
        params = self.main_path(feat)
        gate = self.contrast_gate(feat)
        return params * gate


# --- HGSA v13 Smart Orchestra ---


class HGSA_v13(nn.Module):  # Ваша текущая версия

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_gaussians=7,
                 g_channels=5):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.g_channels = g_channels

        HIDDEN_FEAT = 32

        # 1. Модуль контекстных признаков
        self.encoder = Encoder2D_v13(
            in_channels=in_channels,
            out_channels=HIDDEN_FEAT,
        )

        # 2. Модуль управления экспертами (на базе сырого x)
        self.orchestrator = Orchestrator_v13(
            in_channels=in_channels,
            hidden_channels=HIDDEN_FEAT,
            out_channels=g_channels,
            num_gaussians=num_gaussians,
        )

        # Подготовительная сеть для Гауссиан
        self.xi_net = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, g_channels, 1),
            nn.Sigmoid(),
        )

        # Головы экспертов
        self.expert_heads = nn.ModuleList([
            HyperFFN_Head(HIDDEN_FEAT, HIDDEN_FEAT, g_channels)
            for _ in range(num_gaussians)
        ])
        # Масштаб амплитуды (уже обсудили)
        self.w_scales = nn.Parameter(torch.ones(num_gaussians) * 1.2)
        # Смещение центров (помогает распределить экспертов по гистограмме)
        # Инициализируем равномерно от 0.1 до 0.9
        self.mu_offsets = nn.Parameter(torch.linspace(0.1, 0.9, num_gaussians))
        # Базовая эластичность (насколько эксперт "широкий" по умолчанию)
        self.sigma_bases = nn.Parameter(torch.ones(num_gaussians) * 0.05)        

        # Модуль финальной шлифовки (Chi-Net)
        self.chi_net = ChiNet_v13(
            in_channels=in_channels,
            feat_channels=g_channels,
            hidden_channels=HIDDEN_FEAT,
            out_channels=out_channels,
        )

        # Aux-loss module
        self.usgs_to_img = nn.Conv2d(
            g_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Потоки признаков
        xi = self.xi_net(x)
        feat = self.encoder(x)

        # Оркестрация (маски и смещения)
        gates_normed, taus_all = self.orchestrator(x)

        psi_total = torch.zeros_like(xi)
        p_list = []

        for i in range(self.num_gaussians):
            gate = gates_normed[:, i]
            tau = taus_all[:, i]

            p_e = self.expert_heads[i](feat).view(B, -1, 3, H, W)

            # 1. Амплитуда: Глобальный масштаб * Локальное решение
            w = self.w_scales[i] * torch.tanh(p_e[:, :, 0, :, :])
            
            # 2. Центр: Глобальное смещение + Локальная подстройка
            # Используем clamp, чтобы mu не вылетало за [0, 1]
            mu_local = torch.tanh(p_e[:, :, 1, :, :]) * 0.1 # Локальный сдвиг +- 0.1
            mu = torch.clamp(self.mu_offsets[i] + mu_local, 0.0, 1.0)
            
            # 3. Эластичность и Sigma
            elasticity = torch.sigmoid(p_e[:, :, 2, :, :] + tau) * 1.2
            dist = torch.abs(xi - mu)
            
            # Sigma: Глобальная база + Динамическая эластичность
            # Это дает эксперту "минимальную специализацию"
            curr_base_sigma = torch.abs(self.sigma_bases[i])
            sigma = curr_base_sigma + elasticity * dist 
            
            diff = dist / (sigma + 1e-6)
            psi_i = gate * w * torch.exp(-0.5 * torch.pow(diff, 2))
            psi_total = psi_total + psi_i

            if self.training:
                p_list.append(torch.stack([w, mu, sigma, gate, tau], dim=2))

        # Финальная сборка (Chi-Net)
        # Вычисление промежуточного и финального выходов
        usgs_out = x + self.usgs_to_img(psi_total)

        # Финальный рендеринг через Chi-Net
        main_out = self.chi_net(x, psi_total)

        return (main_out, usgs_out, p_list) if self.training else main_out
