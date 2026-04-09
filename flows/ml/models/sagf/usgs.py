import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualContextBlock(nn.Module):
    """Блок с механизмом Global Context (GCNet)."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.channel_add_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LayerNorm([in_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        context = self.channel_add_conv(out)
        return self.act(out + context + residual)

class FullSizeParameterEncoder(nn.Module):
    """Декодер для полноразмерных карт параметров."""
    def __init__(self, out_dim, in_channels=64):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            ResidualContextBlock(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            ResidualContextBlock(128),
        )
        self.up = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(16, out_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        feat = self.down(x)
        return self.up(feat)


class USGS(nn.Module):
    def __init__(self, M=12, Q=7):
        super().__init__()
        self.M = M  # Количество Гауссиан (внутренняя сумма)
        self.Q = Q  # Количество адаптивных узлов (внешняя сумма)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.SiLU(),
        )

        # 1. Энкодеры параметров SAGF (амплитуда, центр, сигма)
        self.enc_amplitude = FullSizeParameterEncoder(out_dim=M)
        self.enc_center = FullSizeParameterEncoder(out_dim=M)
        self.enc_sigma = FullSizeParameterEncoder(out_dim=M)

        # 2. Адаптивная сетка узлов сканирования (q_nodes)
        # Генерирует Q значений узлов на основе глобального контекста изображения
        self.node_generator = nn.Sequential(
            ResidualContextBlock(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, Q),
            nn.Sigmoid() # Узлы всегда в диапазоне [0, 1]
        )

        # 3. Оценка базовой освещенности
        self.illum_estimator = nn.Sequential(
            ResidualContextBlock(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )

        # 4. Внешняя суперпозиция (chi_q)
        # Используем свертку 1x1 с группами, чтобы эффективно реализовать Q разных функций chi
        self.chi_superposition = nn.Sequential(
            nn.Conv2d(Q * 3, Q * 9, kernel_size=1, groups=Q),
            nn.SiLU(),
            nn.Conv2d(Q * 9, 3, kernel_size=1),  # Финальное смешивание в RGB
        )

        self.psi_norm = nn.InstanceNorm2d(Q * 3)

    def forward(self, x, train=False):
        b, c, h, w = x.shape
        feat = self.stem(x)

        # --- ГЕНЕРАЦИЯ ПАРАМЕТРОВ ---
        a = torch.tanh(self.enc_amplitude(feat)).unsqueeze(1) # [B, 1, M, H, W]
        mu = torch.sigmoid(self.enc_center(feat)).unsqueeze(1) # [B, 1, M, H, W]
        sigma = F.softplus(self.enc_sigma(feat)) + 0.1
        sigma = sigma.unsqueeze(1)

        # Генерируем адаптивные узлы сканирования: [B, Q] -> [B, 1, 1, Q, 1, 1]
        q_nodes = self.node_generator(feat).view(b, 1, 1, self.Q, 1, 1)
        # Сортируем узлы для сохранения топологического порядка яркости
        q_nodes, _ = torch.sort(q_nodes, dim=3)

        # --- БАЗОВАЯ КОРРЕКЦИЯ ---
        illum = self.illum_estimator(feat)
        gain = (torch.sigmoid(illum[:, 0:1]) + 0.5).view(b, 1, 1, 1)
        bias = (torch.tanh(illum[:, 1:2]) * 0.5).view(b, 1, 1, 1)
        x_adj = x * gain + bias

        # --- ВНУТРЕННЯЯ СУММА (SAGF) ---
        # Вычисляем значение поля в адаптивных узлах q_nodes
        # mu: [B, 1, M, 1, H, W], q_nodes: [B, 1, 1, Q, 1, 1]
        diff = (q_nodes - mu.unsqueeze(3)) ** 2
        gaussians = a.unsqueeze(3) * torch.exp(-diff / (2 * sigma.unsqueeze(3)**2))

        # S_M_q: [B, 1, Q, H, W] (значение поля в каждом адаптивном узле)
        S_M_q = torch.sum(gaussians, dim=2)

        # --- ВНЕШНЯЯ СУПЕРПОЗИЦИЯ ---
        # Модулируем входное изображение значениями поля в узлах
        # Подготавливаем тензор: для каждого узла Q создаем 3 канала RGB
        # chi_input: [B, Q*3, H, W]
        chi_input = (S_M_q * x_adj.unsqueeze(2)).view(b, self.Q * 3, h, w)
        chi_input = self.psi_norm(chi_input)

        color_residual = self.chi_superposition(chi_input)

        # Финальный результат
        out = x_adj + color_residual
        out = torch.clamp(out, 0.0, 1.0)

        if train:
            return out, a, mu, sigma

        return out
