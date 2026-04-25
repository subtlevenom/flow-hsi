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
        self.act_layer = nn.SiLU(inplace=True)

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

class HGSA_v5(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_gaussians=8):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.channels = in_channels
        HIDDEN_FEAT = 64

        # 1. Энкодер признаков
        self.encoder = Encoder2D(in_channels, out_dim=HIDDEN_FEAT)

        # 2. Независимые головы для каждой гауссианы
        self.param_heads = nn.ModuleList([
            FFN(HIDDEN_FEAT, HIDDEN_FEAT // 2, 3 * in_channels)
            for _ in range(num_gaussians)
        ])

        # 3. Проекция xi (может быть ненормированной)
        self.xi_proj = FFN(in_channels, in_channels * 2, in_channels)

        # 4. Финальный блок Chi (принимает x и psi_total)
        # Исправлено: убрана опечатка 'ac', структура приведена в порядок
        self.chi_net = FFN(in_channels * 2, HIDDEN_FEAT, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        feat = self.encoder(x)
        xi = self.xi_proj(x)

        psi_total = torch.zeros_like(xi)
        
        for i in range(self.num_gaussians):
            # Параметры p: [B, C*3, H, W]
            p = self.param_heads[i](feat)
            p = p.view(B, C, 3, H, W)

            w = torch.tanh(p[:, :, 0, :, :])
            
            # mu в диапазоне [-3.0, 3.0] для ненормированного xi
            mu = torch.tanh(p[:, :, 1, :, :]) * 3.0
            
            # sigma в диапазоне [0.1, 2.5]
            sigma = torch.sigmoid(p[:, :, 2, :, :]) * 2.4 + 0.1

            # Вычисление без eps, так как sigma >= 0.1
            diff = (xi - mu) / sigma
            psi_total = psi_total + w * torch.exp(-0.5 * diff**2)

        # Конкатенация 3 каналов оригинала и 3 каналов суммы гауссиан
        combined = torch.cat([x, psi_total], dim=1) 
        out = self.chi_net(combined)

        return out