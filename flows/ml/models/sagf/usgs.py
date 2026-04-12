import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from flows.ml.layers.encoders.sg_encoder import LayerNorm
from flows.ml.layers.mw_isp.dwt import DWTForward
from flows.ml.models.ggpd.gpd import create_encoder

# --- Базовые компоненты cmKAN ---

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# class AddCoords(nn.Module):
    # def forward(self, x):
        # b, _, h, w = x.shape
        # y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        # x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        # return torch.cat([x, x_coords, y_coords], dim=1)

class LayerNorm(nn.Module):

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Gated(nn.Module):
    """Блок агрегации признаков в стиле cmKAN: DW-Conv + Gated Expansion."""
    def __init__(self, in_channels:int):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.norm = nn.InstanceNorm2d(in_channels)
        self.pw_expansion = nn.Conv2d(
            in_channels,
            in_channels * 2,
            kernel_size=1,
        )
        self.pw_projection = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
        )
        self.act = nn.SiLU()

    def forward(self, x):
        res = x
        x = self.norm(self.dw_conv(x))
        x = self.pw_expansion(x)
        phi, gate = x.chunk(2, dim=1)
        x = phi * torch.sigmoid(gate)
        x = self.pw_projection(x)
        return x + res

class FFN(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """

    def __init__(self, in_channels, hidden_channels=None, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.pointwise1 = nn.Conv2d(in_channels,
                                    hidden_channels,
                                    kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_channels,
                                   hidden_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   dilation=1,
                                   groups=hidden_channels)
        self.pointwise2 = nn.Conv2d(hidden_channels,
                                    out_channels,
                                    kernel_size=1)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x


class IlluminationEstimator(nn.Module):

    def __init__(self, in_channels=3, mid_channels = 12, out_channels=3):
        super(IlluminationEstimator, self).__init__()

        self.conv = nn.Conv2d(
            in_channels + 1,
            mid_channels,
            kernel_size=1,
            bias=True,
        )
        self.act = nn.SiLU()
        self.fea = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=out_channels,
        )
        self.map = nn.Sequential(
            Gated(mid_channels),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([x, x_mean], dim=1)
        x = self.act(self.conv(x))
        x_fea = self.fea(x)
        x_map = self.map(x_fea)
        return x_fea, x_map


class IlluminationEncoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        N = 4

        # Локальный Illumination Estimator (Gain/Bias карта)
        self.illum_estimator = IlluminationEstimator(in_channels)

        # illumination down 1
        n_channels = in_channels * N
        self.illum_map_down_1 = DWTForward()  # B,4*C,H/2,W/2
        self.illum_fea_down_1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=1,
            ),
        )
        self.illum_trans_1 = create_encoder(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            alg='msab',
            num_blocks = [1],
        )

        # illumination down 2
        n_n_channels = in_channels * N * N
        self.illum_map_down_2 = DWTForward()  # B,16*C,H/4,W/4
        self.illum_fea_down_2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(
                n_channels,
                n_n_channels,
                kernel_size=1,
            ),
        )
        self.illum_trans_2 = create_encoder(
            in_channels=2 * n_n_channels,
            out_channels=n_n_channels,
            alg='msab',
            num_blocks = [1],
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear')

        sum_channels = in_channels * (1 + N + N * N)
        self.conv_out = nn.Sequential(
            LayerNorm(sum_channels),
            FFN(sum_channels, out_channels=out_channels))

    def forward(self, x):
        illum_fea, illum_map = self.illum_estimator(x)

        illum_map = self.illum_map_down_1(illum_map)
        illum_fea = self.illum_fea_down_1(illum_fea)
        x1 = torch.cat([illum_map, illum_fea], dim=1)
        x1 = self.illum_trans_1(x1)

        illum_map = self.illum_map_down_2(x1)
        illum_fea = self.illum_fea_down_2(illum_fea)
        x2 = torch.cat([illum_map, illum_fea], dim=1)
        x2 = self.illum_trans_2(x2)

        x1 = self.up1(x1)
        x2 = self.up2(x2)
        x = torch.cat([x, x1, x2], dim=1)
        x = self.conv_out(x)

        return x

class ParamsEncoder(nn.Module):
    """Специализированный энкодер для параметров (a, mu, sigma)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local_path = nn.Sequential(
            Gated(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=None, size=None, mode='bilinear', align_corners=False)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        l_feat = self.local_path(x)
        # Динамический upsample под размер входа
        g_feat = F.interpolate(self.global_path[0:3](x), size=(h, w), mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([l_feat, g_feat], dim=1))


class USGSEncoder(nn.Module):

    def __init__(self,in_channels:int, mid_channels:int, M:int, Q:int):
        super().__init__()

        self.M = M
        self.Q = Q

        # Локальный Illumination Estimator (Gain/Bias карта)
        self.illum_block = IlluminationEncoder(
            in_channels=in_channels,
            out_channels=mid_channels,
        )

        # Индивидуальные cmKAN-энкодеры для каждого типа параметров
        self.enc_amplitude = ParamsEncoder(mid_channels, M)
        self.enc_center = ParamsEncoder(mid_channels, M)
        self.enc_sigma = ParamsEncoder(mid_channels, M)

        # Адаптивные узлы
        self.node_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid_channels, mid_channels),
            nn.SiLU(),
            nn.Linear(mid_channels, 3 * Q),
            nn.Sigmoid()
        )

    def forward(self, x, train=False):
        B, C, H, W = x.shape

        # illimination block
        feat = self.illum_block(x)

        # 1. Генерация параметров через cmKAN-энкодеры
        a = torch.tanh(self.enc_amplitude(feat)).unsqueeze(1).unsqueeze(1)
        # Ограничиваем mu [-0.5, 1.5] для стабильности Гауссиан
        mu = (torch.tanh(self.enc_center(feat)) + 0.5).unsqueeze(1).unsqueeze(1)
        # Ограничиваем sigma [0.05, 0.5] для стабильности Гауссиан
        sigma = (torch.sigmoid(self.enc_sigma(feat)) * 0.95 + 0.05).unsqueeze(1).unsqueeze(1)

        # 2. Адаптивные узлы
        q_nodes = self.node_generator(feat).view(B, C, self.Q, 1, 1, 1)
        q_nodes, _ = torch.sort(q_nodes, dim=2)
        # this wors only if in_channels == out_channels
        x = x.view(B, C, 1, 1, H, W)
        q_nodes = x * q_nodes

        # B C Q M H W
        return q_nodes, a, mu, sigma

# --- Основная архитектура ---

# class HC_USGS_CMKAN(nn.Module):
class USGS(nn.Module):
    def __init__(self,in_channels:int, out_channels:int, M:int=12, Q:int=7,):
        super().__init__()
        self.M = M
        self.Q = Q

        MID_CHANNELS = 21 * in_channels

        self.usgs_encoder = USGSEncoder(
            in_channels=in_channels,
            mid_channels=MID_CHANNELS,
            M=M,
            Q=Q,
        )

        self.layer_norm = LayerNorm(3)
        self.psi_norm = nn.InstanceNorm2d(Q * 3)

        # Углубленный блок суперпозиции (Chi)
        self.chi_superposition = nn.Sequential(
            nn.Conv2d(Q * 3 + 6, Q * 6, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(Q * 6, Q * 6, kernel_size=3, padding=1, groups=Q),
            nn.SiLU(),
            nn.Conv2d(Q * 6, 3, kernel_size=1)
        )


    def forward(self, x:torch.Tensor, train=False):
        B, C, H, W = x.shape

        # 1. encoder block
        q, a, mu, sigma = self.usgs_encoder(x)

        # 2. Вычисление USGS поля
        # diff: [B, C, Q, M, H, W]
        diff = (q - mu) ** 2
        gaussians = a * torch.exp(-diff / (2 * sigma**2))
        S_M_q = torch.sum(gaussians, dim=3) # [B, C, Q, H, W]

        # 5. Суперпозиция (Chi)
        # x = self.coord_adder(x)
        chi_input = rearrange(S_M_q, 'b c q h w -> b (c q) h w')
        chi_input = self.psi_norm(chi_input)

        x_norm = self.layer_norm(x)
        chi_input = torch.cat([chi_input, x, x_norm], dim=1)

        y = self.chi_superposition(chi_input)
        y = rearrange(y, 'b (c q) h w -> b c q h w', c=C)

        out = torch.sum(y, dim=2)
        out = torch.clamp(out, 0.0, 1.0)

        if train:
            return out, a, mu, sigma
        return out
