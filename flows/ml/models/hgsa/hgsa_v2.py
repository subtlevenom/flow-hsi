import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. ГЛОБАЛЬНЫЙ ЦВЕТОВОЙ ЭНКОДЕР (Cross-Scale с динамическим V) ---

class GlobalColorEncoder(nn.Module):
    def __init__(self, channels=64, hidden=64, pool_scales=(1, 2, 4, 8)):
        super().__init__()
        self.pool_scales = pool_scales
        self.feat_x = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden) # Динамический Value
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, feat):
        b, c, h, w = feat.shape
        f_x = self.feat_x(feat)
        tokens = [F.adaptive_avg_pool2d(f_x, s).view(b, -1, c) for s in self.pool_scales]
        token_seq = torch.cat(tokens, dim=1)
        
        q, k, v = self.query(token_seq), self.key(token_seq), self.value(token_seq)
        attn = (q @ k.transpose(-2, -1)) * (c**-0.5)
        attn = attn.softmax(dim=-1)
        z = (attn @ v).mean(dim=1)
        return self.proj(z).view(b, -1, 1, 1)

# --- 2. PENTAMODAL ATTENTION (Параллельные ветви) ---

class PentamodalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Ветвь 1: Channel (Параллельно)
        self.channel_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        # Ветвь 2: Spatial (Параллельно)
        self.spatial_path = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        # Ветвь 3: Global Context (Параллельно)
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.SiLU()
        )
        # Ветвь 4: Local Detail (Параллельно)
        self.local_path = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.SiLU()
        )
        # Слияние всех параллельных путей
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, x):
        c_attn = x * self.channel_path(x)
        s_attn = x * self.spatial_path(x)
        g_feat = self.global_path(x).expand_as(x)
        l_feat = self.local_path(x)
        
        # Объединение (Identity + Внимания + Глобальное + Локальное)
        combined = torch.cat([c_attn + s_attn, g_feat, l_feat], dim=1)
        return self.fusion(combined) + x

# --- 3. ОДНОСЛОЙНОЕ ГАУССОВО ЯДРО ---

class SingleLayerGaussianCore(nn.Module):
    def __init__(self, in_channels, M=32):
        super().__init__()
        self.M = M
        self.hyper = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, M * 3, kernel_size=1)
        )

    def forward(self, feat, guide):
        b, c, h, w = feat.shape
        params = self.hyper(feat)
        a, mu, sigma = torch.chunk(params, 3, dim=1)
        
        a = torch.tanh(a)
        mu = torch.sigmoid(mu)
        sigma = torch.sigmoid(sigma) * 0.4 + 0.1
        
        grid = guide.mean(dim=1, keepdim=True)
        diff = (grid.unsqueeze(1) - mu.unsqueeze(2))**2
        gaussians = a.unsqueeze(2) * torch.exp(-diff / (2 * sigma.unsqueeze(2)**2 + 1e-6))
        return torch.sum(gaussians, dim=1)

# --- 4. ГИБРИДНАЯ ГОЛОВА (Матричный трансформ) ---

class HybridColorHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.matrix_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 12, kernel_size=1),
        )
        self.residual_gate = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, feat, src):
        params = self.matrix_head(feat)
        b, _, h, w = params.shape
        m_delta = torch.tanh(params[:, :9]).view(b, 3, 3, h, w)
        bias = torch.tanh(params[:, 9:12]).view(b, 3, h, w) * 0.1
        
        eye = torch.eye(3, device=src.device).view(1, 3, 3, 1, 1)
        matrix = eye + 0.1 * m_delta
        transformed = torch.einsum("boihw,bihw->bohw", matrix, src) + bias
        
        res = self.residual_gate(feat)
        return torch.clamp(transformed + 0.1 * res, 0.0, 1.0)

# --- 5. ФИНАЛЬНАЯ МОДЕЛЬ HGSA v2 Pentamodal ---

class HGSA_v2(nn.Module):
    def __init__(self, hidden=64, M=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden),
            nn.SiLU()
        )
        self.global_encoder = GlobalColorEncoder(channels=hidden, hidden=hidden)
        self.pentamodal_processor = PentamodalAttention(hidden)
        self.gaussian_core = SingleLayerGaussianCore(in_channels=hidden, M=M)
        self.head = HybridColorHead(in_channels=hidden)

    def forward(self, src):
        feat = self.stem(src)
        
        # 1. Глобальный контекст (Cross-Scale)
        color_context = self.global_encoder(feat)
        feat = feat * torch.sigmoid(color_context)
        
        # 2. Pentamodal анализ (Параллельные ветви)
        feat = self.pentamodal_processor(feat)
        
        # 3. Однослойная Гауссова модуляция
        mod_map = self.gaussian_core(feat, src)
        feat = feat * mod_map
        
        # 4. Рендеринг
        return self.head(feat, src)