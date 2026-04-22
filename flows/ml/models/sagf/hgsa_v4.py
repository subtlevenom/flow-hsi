import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --- Базовые утилиты (без изменений для совместимости) ---
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

    def __init__(self):
        super(DWTForward, self).__init__()

    def forward(self, x):
        return F.pixel_unshuffle(x, 2)


# --- Компоненты cmKAN ---
class IlluminationEstimator(nn.Module):

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(IlluminationEstimator, self).__init__()
        self.conv1 = nn.Conv2d(
            n_fea_in,
            n_fea_middle,
            kernel_size=1,
            bias=True,
        )
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_in,
        )
        self.conv2 = nn.Conv2d(
            n_fea_middle,
            n_fea_out,
            kernel_size=1,
            bias=True,
        )

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
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illu_feat):
        b, c, h, w = x.shape
        q, k, a = self.q_proj(x), self.k_proj(x), self.a_proj(x)
        v = self.v_proj(x) * illu_feat
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
        out = rearrange(attn_a @ (attn_k @ v),
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
        self.act_layer = nn.SiLU(
            inplace=True)  # Заменил на SiLU для лучших градиентов

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


class Encoder2D(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Encoder2D, self).__init__()
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


# --- 4. Финальная модель HGSA_v4 (Spatial USGS) ---


class HGSA_v4(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, num_gaussians=16):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.channels = in_channels
        HIDDEN_FEAT = 64

        # 1. Тяжелый энкодер (cmKAN)
        self.encoder = Encoder2D(in_channels, out_dim=HIDDEN_FEAT)

        # 2. Пространственно-зависимый генератор параметров (Spatial-Variant)
        # Выход: [B, num_gaussians * 3, H, W]
        self.param_head = nn.Sequential(
            nn.Conv2d(HIDDEN_FEAT, HIDDEN_FEAT, kernel_size=3, padding=1),
            nn.SiLU(), nn.Conv2d(HIDDEN_FEAT, num_gaussians * 3,
                                 kernel_size=1))

        # 3. Внутренняя проекция xi
        self.xi_proj = FFN(in_channels, in_channels * 2, in_channels)

        # 4. Внешняя функция Chi (Gated Convolutional FFN)
        # Реализуем через свертки для учета контекста
        self.chi_gate = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  padding=1)
        self.chi_feat = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  padding=1)
        self.chi_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 5. Параметры Шпрехера
        self.lambdas = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # А. Извлечение признаков и генерация пространственных параметров
        feat = self.encoder(x)
        # params: [B, G*3, H, W]
        params = self.param_head(feat)
        params = params.view(B, self.num_gaussians, 3, H, W)

        # Б. Подготовка xi
        xi = self.xi_proj(x)

        # В. Сумма Гауссиан (Spatial USGS)
        psi_total = torch.zeros_like(xi)
        for i in range(self.num_gaussians):
            p = params[:, i, :, :, :]  # [B, 3, H, W]

            w = torch.tanh(p[:, 0:1, :, :])
            # mu в диапазоне [-0.5, 1.5]
            mu = torch.sigmoid(p[:, 1:2, :, :]) * 2.0 - 0.5
            # sigma ограничена снизу 0.01 и сверху 0.51
            sigma = torch.sigmoid(p[:, 2:3, :, :]) * 2.0 + 0.1

            diff = (xi - mu) / sigma
            psi_total = psi_total + w * torch.exp(-0.5 * diff**2)

        # Г. Внешняя функция Chi (Gated Convolutional)
        # Используем пространственную информацию вместо попиксельного Linear
        modulated = psi_total * self.lambdas
        gate = torch.sigmoid(self.chi_gate(modulated))
        feat_chi = F.silu(self.chi_feat(modulated))
        h = self.chi_out(gate * feat_chi)

        # Финальный выход с Residual
        return self.final_conv(h + x)
