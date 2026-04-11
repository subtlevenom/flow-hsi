import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ВСПОМОГАТЕЛЬНЫЕ БЛОКИ ---

class AddCoords(nn.Module):
    """Добавляет нормализованные x, y координаты для учета пространственного контекста."""
    def forward(self, x):
        b, _, h, w = x.shape
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, x_coords, y_coords], dim=1)

class GatedConvBlock(nn.Module):
    """Блок со стробированием (Gate) для фильтрации цветовых признаков."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x_gated = self.conv(x)
        phi, gate = x_gated.chunk(2, dim=1)
        out = phi * torch.sigmoid(gate)
        return self.proj(out) + x

class ResidualContextBlock(nn.Module):
    """Блок глобального контекста (GCNet style)."""
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
        out = self.conv(x)
        context = self.channel_add_conv(out)
        return self.act(out + context + x)

# --- УЛУЧШЕННЫЙ ЭНКОДЕР ПАРАМЕТРОВ ---

class AdvancedParameterEncoder(nn.Module):
    """Энкодер параметров в стиле cmKAN: Global + Local ветки."""
    def __init__(self, out_dim, in_channels=64):
        super().__init__()
        # Глобальная ветка (понимает освещение всего кадра)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        )
        
        # Локальная ветка (реагирует на текстуры и границы)
        self.local_branch = nn.Sequential(
            GatedConvBlock(in_channels),
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU()
        )

        # Агрегатор
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_dim, kernel_size=1)
        )

    def forward(self, x):
        g_feat = self.global_branch(x)
        l_feat = self.local_branch(x)
        return self.fusion(torch.cat([g_feat, l_feat], dim=1))

# --- ОСНОВНАЯ МОДЕЛЬ HC-USGS ---

class USGS(nn.Module):
    def __init__(self, M=12, Q=7):
        super().__init__()
        self.M = M  # Количество Гауссиан
        self.Q = Q  # Количество адаптивных узлов сканирования

        self.coord_adder = AddCoords()
        
        # Stem: принимает RGB + 2 канала координат
        self.stem = nn.Sequential(
            nn.Conv2d(3 + 2, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.SiLU(),
        )

        # Гиперсети для параметров SAGF
        self.enc_amplitude = AdvancedParameterEncoder(out_dim=M)
        self.enc_center = AdvancedParameterEncoder(out_dim=M)
        self.enc_sigma = AdvancedParameterEncoder(out_dim=M)

        # Генератор адаптивной сетки узлов
        self.node_generator = nn.Sequential(
            ResidualContextBlock(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, Q),
            nn.Sigmoid()
        )

        # Оценка базовой экспозиции (Gain/Bias)
        self.illum_estimator = nn.Sequential(
            ResidualContextBlock(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )

        # Блок внешней суперпозиции (chi)
        # Используем группы Q для разделения функций активации каждого узла
        self.chi_superposition = nn.Sequential(
            nn.Conv2d(Q * 3, Q * 9, kernel_size=3, padding=1, groups=Q),
            nn.SiLU(),
            nn.Conv2d(Q * 9, 3, kernel_size=1)
        )

        self.psi_norm = nn.InstanceNorm2d(Q * 3)

    def forward(self, x, train=False):
        b, c, h, w = x.shape
        
        # 1. Извлечение признаков с учетом координат
        x_coords = self.coord_adder(x)
        feat = self.stem(x_coords)

        # 2. Генерация параметров Гауссова поля
        a = torch.tanh(self.enc_amplitude(feat)).unsqueeze(1) # [B, 1, M, H, W]
        mu = torch.sigmoid(self.enc_center(feat)).unsqueeze(1) # [B, 1, M, H, W]
        sigma = F.softplus(self.enc_sigma(feat)) + 0.1
        sigma = sigma.unsqueeze(1)

        # 3. Адаптивные узлы сканирования
        q_nodes = self.node_generator(feat).view(b, 1, 1, self.Q, 1, 1)
        q_nodes, _ = torch.sort(q_nodes, dim=3) # Сохраняем порядок яркости

        # 4. Базовая коррекция (Global Gain/Bias)
        illum = self.illum_estimator(feat)
        gain = (torch.sigmoid(illum[:, 0:1]) + 0.5).view(b, 1, 1, 1)
        bias = (torch.tanh(illum[:, 1:2]) * 0.5).view(b, 1, 1, 1)
        x_adj = x * gain + bias

        # 5. Внутренняя сумма (SAGF): вычисление поля в узлах q_nodes
        # diff: [B, 1, M, Q, H, W]
        diff = (q_nodes - mu.unsqueeze(3)) ** 2
        gaussians = a.unsqueeze(3) * torch.exp(-diff / (2 * sigma.unsqueeze(3)**2))
        
        # S_M_q: [B, 1, Q, H, W]
        S_M_q = torch.sum(gaussians, dim=2)

        # 6. Внешняя суперпозиция (chi)
        # Модулируем x_adj значениями поля в узлах и объединяем в каналы
        chi_input = (S_M_q * x_adj.unsqueeze(2)).view(b, self.Q * 3, h, w)
        chi_input = self.psi_norm(chi_input)
        
        color_residual = self.chi_superposition(chi_input)

        # 7. Финальная аддитивная сборка
        out = x_adj + color_residual
        out = torch.clamp(out, 0.0, 1.0)

        if train:
            return out, a, mu, sigma

        return out