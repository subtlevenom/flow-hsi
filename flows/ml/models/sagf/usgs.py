import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Базовые компоненты cmKAN ---

class AddCoords(nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, x_coords, y_coords], dim=1)

class CMKANBlock(nn.Module):
    """Блок агрегации признаков в стиле cmKAN: DW-Conv + Gated Expansion."""
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.InstanceNorm2d(dim)
        self.pw_expansion = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.pw_projection = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        res = x
        x = self.norm(self.dw_conv(x))
        x = self.pw_expansion(x)
        phi, gate = x.chunk(2, dim=1)
        x = phi * torch.sigmoid(gate)
        x = self.pw_projection(x)
        return x + res

class CMKANEncoder(nn.Module):
    """Специализированный энкодер для параметров (a, mu, sigma)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local_path = nn.Sequential(
            CMKANBlock(in_channels),
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

# --- Основная архитектура ---

# class HC_USGS_CMKAN(nn.Module):
class USGS(nn.Module):
    def __init__(self, M=12, Q=7):
        super().__init__()
        self.M = M
        self.Q = Q

        self.coord_adder = AddCoords()
        self.stem = nn.Sequential(
            nn.Conv2d(3 + 2, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.SiLU()
        )

        # Индивидуальные cmKAN-энкодеры для каждого типа параметров
        self.enc_amplitude = CMKANEncoder(64, M)
        self.enc_center = CMKANEncoder(64, M)
        self.enc_sigma = CMKANEncoder(64, M)

        # Локальный Illumination Estimator (Gain/Bias карта)
        self.illum_estimator = nn.Sequential(
            CMKANBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 2, kernel_size=1)
        )

        # Адаптивные узлы
        self.node_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, Q),
            nn.Sigmoid()
        )

        # Углубленный блок суперпозиции (Chi)
        self.chi_superposition = nn.Sequential(
            nn.Conv2d(Q * 3, Q * 6, kernel_size=1, groups=Q),
            nn.SiLU(),
            nn.Conv2d(Q * 6, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        
        self.psi_norm = nn.InstanceNorm2d(Q * 3)

    def forward(self, x, train=False):
        b, c, h, w = x.shape
        
        feat = self.stem(self.coord_adder(x))

        # 1. Генерация параметров через cmKAN-энкодеры
        a = torch.tanh(self.enc_amplitude(feat)).unsqueeze(1) 
        mu = torch.sigmoid(self.enc_center(feat)).unsqueeze(1)
        # Ограничиваем sigma [0.05, 0.5] для стабильности Гауссиан
        sigma = (torch.sigmoid(self.enc_sigma(feat)) * 0.45 + 0.05).unsqueeze(1)

        # 2. Локальная коррекция яркости (Gain/Bias карты)
        illum = self.illum_estimator(feat)
        gain = torch.sigmoid(illum[:, 0:1, :, :]) + 0.5
        bias = torch.tanh(illum[:, 1:2, :, :]) * 0.5
        x_adj = x * gain + bias

        # 3. Адаптивные узлы
        q_nodes = self.node_generator(feat).view(b, 1, 1, self.Q, 1, 1)
        q_nodes, _ = torch.sort(q_nodes, dim=3)

        # 4. Вычисление USGS поля
        # diff: [B, 1, M, Q, H, W]
        diff = (q_nodes - mu.unsqueeze(3)) ** 2
        gaussians = a.unsqueeze(3) * torch.exp(-diff / (2 * sigma.unsqueeze(3)**2))
        S_M_q = torch.sum(gaussians, dim=2) # [B, 1, Q, H, W]

        # 5. Суперпозиция (Chi)
        chi_input = (S_M_q * x_adj.unsqueeze(2)).view(b, self.Q * 3, h, w)
        chi_input = self.psi_norm(chi_input)
        color_residual = self.chi_superposition(chi_input)

        out = x_adj + color_residual
        out = torch.clamp(out, 0.0, 1.0)

        if train:
            return out, a, mu, sigma
        return out