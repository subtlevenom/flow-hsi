import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualContextBlock(nn.Module):
    """Блок с механизмом Global Context (GCNet) для извлечения глобальных признаков."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # Global Context Modeling
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


class ParameterEncoder(nn.Module):
    """Специализированный энкодер для конкретного параметра (a, mu или sigma)."""

    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            ResidualContextBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 1/2
            ResidualContextBlock(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TrainableNodes(nn.Module):

    def __init__(self, num_nodes):
        super().__init__()
        # Инициализируем равномерные интервалы
        init_deltas = torch.ones(num_nodes) / num_nodes
        self.deltas = nn.Parameter(init_deltas)

    def forward(self):
        # Гарантируем положительность шага и нормализуем, чтобы сумма была 2 (диапазон от -1 до 1)
        normalized_deltas = F.softplus(self.deltas)
        normalized_deltas = 2.0 * normalized_deltas / normalized_deltas.sum()

        # Вычисляем позиции узлов через кумулятивную сумму
        nodes = torch.cumsum(normalized_deltas, dim=0) - 1.0
        return nodes


class USGS(nn.Module):
    def __init__(self, M=12, num_nodes=32):
        super().__init__()
        self.M = M
        self.num_nodes = num_nodes

        # Общий входной блок (Stem)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.SiLU(),
        )

        # 1. Раздельные энкодеры параметров
        self.enc_amplitude = ParameterEncoder(M)
        self.enc_center = ParameterEncoder(M)
        self.enc_sigma = ParameterEncoder(M)

        # 2. Блок оценки освещенности (Illumination Estimator)
        # Оценивает глобальный коэффициент экспозиции и гамму
        self.illum_estimator = nn.Sequential(
            ResidualContextBlock(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 2),  # [gain, gamma]
        )

        # 3. Внешняя активация chi (Superposition)
        self.chi = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))

        # self.register_buffer('q_nodes', torch.linspace(-1, 1, num_nodes))
        self.node_generator = TrainableNodes(num_nodes)

    def forward(self, x, train=False):
        b, c, h, w = x.shape
        feat = self.stem(x)

        _a = self.enc_amplitude(feat)  # [B, M, 1]
        _mu = self.enc_center(feat)  # [B, M, 1]
        # _sigma = torch.exp(self.enc_sigma(feat)) + 1e-4
        _sigma = F.softplus(self.enc_sigma(feat)) + 1e-5

        # Извлечение параметров
        a = _a.unsqueeze(-1)  # [B, M, 1]
        mu = _mu.unsqueeze(-1)  # [B, M, 1]
        sigma = _sigma.unsqueeze(-1)

        # Оценка освещенности
        illum_params = self.illum_estimator(feat)
        # gain = torch.sigmoid(illum_params[:, 0:1]) * 2.0  # Усиление [0, 2]
        gain = torch.tanh(illum_params[:, 0:1])  # Усиление [0, 2]
        # gamma = torch.exp(illum_params[:, 1:2])  # Гамма-коррекция
        gamma = F.softplus(illum_params[:, 1:2])  # Гамма-коррекция

        # USGS Ядро
        # y = self.q_nodes.view(1, 1, -1)
        q_nodes = self.node_generator()  # [num_nodes]
        y = q_nodes.view(1, 1, -1)
        gaussians = a * torch.exp(-((y - mu) ** 2) / (2 * sigma**2))
        psi_q = torch.sum(gaussians, dim=1)  # [B, num_nodes]

        # Суперпозиция
        chi_val = self.chi(psi_q.view(-1, 1)).view(b, self.num_nodes)
        color_factor = torch.sum(chi_val, dim=1).view(b, 1, 1, 1)

        # Финальная трансформация: USGS + Illumination
        # Применяем коррекцию освещенности и USGS-маппинг
        x_adj = torch.pow(F.sigmoid(x + gain.view(b, 1, 1, 1)), gamma.view(b, 1, 1, 1))
        out = x_adj * color_factor

        if train:
            return out, _a, _mu, _sigma

        return out
