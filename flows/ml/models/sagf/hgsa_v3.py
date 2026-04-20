import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Full cmKAN Generator Logic (Адаптировано под Гауссианы) ---
class CM_Gaussian_Generator(nn.Module):
    """
    Реализация генератора из cmKAN: свертки -> пулинг -> головы.
    https://github.com/gosha20777/cmKAN/blob/main/cm_kan/ml/layers/cm_kan/generator.py
    """
    def __init__(self, in_channels, num_gaussians, hidden_dim=64):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # Сверточный блок извлечения признаков (как в оригинале)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        
        # Глобальный агрегатор контекста
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Разделенные головы для каждой Гауссианы (w, mu, sigma)
        # Каждая голова - это маленький MLP, как в оригинальном коде
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 3)
            ) for _ in range(num_gaussians)
        ])

    def forward(self, x):
        b = x.shape[0]
        # Извлекаем глубокие признаки
        feat = self.features(x)
        # Сжимаем в вектор контекста
        ctx = self.global_pool(feat).view(b, -1)
        
        # Генерируем параметры через головы
        params = []
        for head in self.heads:
            p = head(ctx) # [B, 3]
            params.append(p.unsqueeze(1))
            
        return torch.cat(params, dim=1) # [B, G, 3]

# --- 2. Input Projector (Внутренняя функция xi) ---
class CM_InputProjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Используем структуру FFN из cmKAN
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(4, channels)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. Illumination Estimation (из cmKAN) ---
class CM_Illumination(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- 4. Итоговая модель: Gaussian-USGS (Single Layer) ---
class HGSA_v3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_gaussians=12, base_channels=24):
        super().__init__()
        
        # Входной стем
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 1. Полноценный cmKAN Генератор
        self.generator = CM_Gaussian_Generator(base_channels, num_gaussians, hidden_dim=48)
        
        # 2. Оценка освещенности
        self.illum_est = CM_Illumination(base_channels)
        
        # 3. Проектор xi
        self.xi_proj = CM_InputProjector(base_channels)
        
        # 4. Внешняя функция Chi (Gated FFN)
        self.chi_w1 = nn.Linear(base_channels, base_channels * 2)
        self.chi_w2 = nn.Linear(base_channels, base_channels * 2)
        self.chi_out = nn.Linear(base_channels * 2, base_channels)
        
        # Веса Шпрехера
        self.lambdas = nn.Parameter(torch.ones(1, base_channels, 1, 1))
        
        # Выход
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x_in):
        x = self.stem(x_in)
        B, C, H, W = x.shape
        
        # А. Генерация параметров Гауссиан (Full Encoder)
        params = self.generator(x) # [B, G, 3]
        
        # Б. Подготовка сигнала
        illum = self.illum_est(x)
        xi = self.xi_proj(x * illum)
        
        # В. Сумма Гауссиан (USGS)
        psi_total = torch.zeros_like(xi)
        for i in range(params.shape[1]):
            p = params[:, i, :]
            w = torch.tanh(p[:, 0]).view(B, 1, 1, 1)
            mu = torch.sigmoid(p[:, 1]).view(B, 1, 1, 1)
            sigma = (F.softplus(p[:, 2]) + 0.05).view(B, 1, 1, 1)
            
            diff = (xi - mu) / sigma
            psi_total = psi_total + w * torch.exp(-0.5 * diff**2)
            
        # Г. Внешняя функция Chi (Gated)
        combined = (psi_total * self.lambdas).permute(0, 2, 3, 1).reshape(-1, C)
        gate = self.chi_w1(combined)
        feat = self.chi_w2(combined)
        res = self.chi_out(F.silu(gate) * feat)
        
        h = res.view(B, H, W, C).permute(0, 3, 1, 2)
        
        return self.final(h + x)
