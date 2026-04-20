import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --- 1. Базовые утилиты cmKAN ---


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DWTForward(nn.Module):
    """ Имитация DWT через PixelUnshuffle для сохранения логики каналов cmKAN """

    def __init__(self):
        super(DWTForward, self).__init__()

    def forward(self, x):
        return F.pixel_unshuffle(x, 2)


# --- 2. Компоненты cmKAN (Illumination, Attention, FFN) ---


class IlluminationEstimator(nn.Module):

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(IlluminationEstimator, self).__init__()
        self.conv1 = nn.Conv2d(n_fea_in,
                               n_fea_middle,
                               kernel_size=1,
                               bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle,
                                    n_fea_middle,
                                    kernel_size=5,
                                    padding=2,
                                    bias=True,
                                    groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle,
                               n_fea_out,
                               kernel_size=1,
                               bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class Attention(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature_a = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_v = nn.Parameter(torch.ones(num_heads, 1, 1))

        # q, k, a уменьшают разрешение (stride=2)
        self.q_proj = nn.Conv2d(dim,
                                dim,
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                groups=dim,
                                bias=bias)
        self.k_proj = nn.Conv2d(dim,
                                dim,
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                bias=bias)
        self.a_proj = nn.Sequential(
            nn.Conv2d(dim,
                      dim,
                      kernel_size=3,
                      padding=1,
                      stride=2,
                      groups=dim,
                      bias=bias), nn.Conv2d(dim, dim // 2, kernel_size=1))
        # v сохраняет разрешение для перемножения с illu_feat
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illu_feat):
        b, c, h, w = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        a = self.a_proj(x)
        v = self.v_proj(x) * illu_feat  # Размер совпадает (h, w)

        q = rearrange(q,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        a = rearrange(a,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q, k, a = F.normalize(q,
                              dim=-1), F.normalize(k,
                                                   dim=-1), F.normalize(a,
                                                                        dim=-1)

        attn_a = (q @ a.transpose(-2, -1)) * self.temperature_a
        attn_a = attn_a.softmax(dim=-1)
        attn_k = (a @ k.transpose(-2, -1)) * self.temperature_v
        attn_k = attn_k.softmax(dim=-1)

        out_v = (attn_k @ v)
        out = (attn_a @ out_v)
        out = rearrange(out,
                        'b head c (h w) -> b (head c) h w',
                        head=self.num_heads,
                        h=h,
                        w=w)
        return self.project_out(out)


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        self.out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features,
                                    hidden_features,
                                    kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_features,
                                   hidden_features,
                                   kernel_size=3,
                                   padding=1,
                                   groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features,
                                    self.out_features,
                                    kernel_size=1)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.pointwise2(
            self.act_layer(self.depthwise(self.pointwise1(x))))


class TransformerBlock(nn.Module):

    def __init__(self, in_channel, mid_channel, out_channel, num_heads, bias):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(in_channel)
        self.attn = Attention(in_channel, num_heads, bias)
        self.norm2 = LayerNorm(in_channel)
        self.ffn = FFN(in_channel, mid_channel, out_channel)

    def forward(self, x, illu_feat):
        x = x + self.attn(self.norm1(x), illu_feat)
        x = x + self.ffn(self.norm2(x))
        return x


# --- 3. Encoder2D (cmKAN Heavy) ---


class Encoder2D(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Encoder2D, self).__init__()
        self.estimator = IlluminationEstimator(12, in_dim + 1, in_dim)

        self.down1 = DWTForward()  # h -> h/2, c=3 -> 12
        self.trans1 = TransformerBlock(12, 12, 12, 3, True)
        self.illu_down1 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(12, 12, 1))

        self.down2 = DWTForward()  # h/2 -> h/4, c=12 -> 48
        self.trans2 = TransformerBlock(48, 48, 48, 3, True)
        self.illu_down2 = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(12, 48, 1))

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear')

        # Вход в FFN: 3 (orig) + 12 (up1) + 48 (up2) = 63
        self.conv_out = nn.Sequential(
            LayerNorm(in_dim + 12 + 48),
            FFN(in_dim + 12 + 48, out_dim, out_dim)  # Явно фиксируем out_dim
        )

    def forward(self, x):
        illu_fea, illu_map = self.estimator(x)
        x_orig = x * illu_map + x

        x1 = self.down1(x_orig)
        i1 = self.illu_down1(illu_fea)
        x1 = self.trans1(x1, i1)

        x2 = self.down2(x1)
        i2 = self.illu_down2(illu_fea)
        x2 = self.trans2(x2, i2)

        x1_up = self.up1(x1)
        x2_up = self.up2(x2)

        out = torch.cat([x_orig, x1_up, x2_up], dim=1)
        return self.conv_out(out)


# --- 4. Финальная модель Gaussian-USGS (Sprecher Edition) ---


class HGSA_v3(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, num_gaussians=16):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.channels = in_channels

        # 1. Тяжелый энкодер (cmKAN)
        HIDDEN_FEAT = 64
        self.encoder = Encoder2D(in_channels, out_dim=HIDDEN_FEAT)

        # 2. Генератор параметров Гауссиан
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.param_head = nn.Sequential(nn.Linear(HIDDEN_FEAT, 128), nn.SiLU(),
                                        nn.Linear(128, num_gaussians * 3))

        # 3. Внутренняя проекция xi (Шпрехер)
        self.xi_proj = FFN(in_channels, in_channels * 2, in_channels)

        # 4. Внешняя функция Chi (Gated FFN)
        self.chi_w1 = nn.Linear(in_channels, in_channels * 2)
        self.chi_w2 = nn.Linear(in_channels, in_channels * 2)
        self.chi_out = nn.Linear(in_channels * 2, in_channels)

        # 5. Лямбды Шпрехера и финальная коррекция
        self.lambdas = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # А. Извлечение признаков и генерация параметров
        feat = self.encoder(x)  # [B, 64, H, W]
        pooled = self.avg_pool(feat).view(B, -1)  # [B, 64]
        params = self.param_head(pooled).view(B, self.num_gaussians, 3)

        # Б. Подготовка xi
        xi = self.xi_proj(x)

        # В. Сумма Гауссиан (USGS)
        psi_total = torch.zeros_like(xi)
        for i in range(self.num_gaussians):
            p = params[:, i, :]
            w = torch.tanh(p[:, 0]).view(B, 1, 1, 1)
            mu = torch.sigmoid(p[:, 1]).view(B, 1, 1, 1)
            sigma = (F.softplus(p[:, 2]) + 0.05).view(B, 1, 1, 1)

            diff = (xi - mu) / sigma
            psi_total = psi_total + w * torch.exp(-0.5 * diff**2)

        # Г. Внешняя функция Chi (Gated)
        combined = (psi_total * self.lambdas).permute(0, 2, 3,
                                                      1).reshape(-1, C)
        gate = self.chi_w1(combined)
        feat_chi = self.chi_w2(combined)
        res = self.chi_out(F.silu(gate) * feat_chi)

        h = res.view(B, H, W, C).permute(0, 3, 1, 2)

        # Финальный выход с Residual
        return self.final_conv(h + x)
