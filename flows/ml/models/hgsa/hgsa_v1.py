import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. ВСПОМОГАТЕЛЬНЫЕ МОДУЛИ КОНТЕКСТА ---

class AddCoords(nn.Module):
    """Добавляет нормализованные x, y координаты для учета пространственного контекста."""
    def forward(self, x):
        b, _, h, w = x.shape
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, x_coords, y_coords], dim=1)

class GatedConvBlock(nn.Module):
    """Блок со стробированием для фильтрации признаков."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x_gated = self.conv(x)
        phi, gate = x_gated.chunk(2, dim=1)
        out = phi * torch.sigmoid(gate)
        return self.proj(out) + x

# --- 2. ГИПЕРСЕТИ И ЭНКОДЕРЫ ---

class HypernetworkParameterEncoder(nn.Module):
    """Генератор параметров для Гауссовых полей на основе локальных и глобальных признаков."""
    def __init__(self, out_dim, in_channels=64, global_channels=64, hidden=64):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
        )
        self.global_affine = nn.Sequential(
            nn.Conv2d(global_channels, hidden * 2, kernel_size=1),
            nn.Tanh(),
        )
        self.out_head = nn.Conv2d(hidden, out_dim, kernel_size=1)

    def forward(self, feat, global_code):
        f = self.local(feat)
        gamma, beta = torch.chunk(self.global_affine(global_code), 2, dim=1)
        f = (1.0 + 0.5 * gamma) * f + beta
        return self.out_head(f)

class GlobalColorEncoder(nn.Module):
    """Извлечение глобального цветового кода через Cross-Scale Attention."""
    def __init__(self, channels=64, hidden=64, pool_scales=(1, 2, 4, 8)):
        super().__init__()
        self.pool_scales = pool_scales
        self.feat_x = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.token_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, batch_first=True)
        self.token_gate = nn.Linear(hidden, 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, feat):
        f_x = self.feat_x(feat)
        tokens = [F.adaptive_avg_pool2d(f_x, s).mean(dim=(2, 3)) for s in self.pool_scales]
        token_seq = torch.stack(tokens, dim=1)
        token_ctx, _ = self.token_attn(token_seq, token_seq, token_seq)
        token_w = F.softmax(self.token_gate(token_ctx).squeeze(-1), dim=1)
        z = (token_ctx * token_w.unsqueeze(-1)).sum(dim=1)
        return self.proj(z).unsqueeze(-1).unsqueeze(-1)

# --- 3. ГИБРИДНЫЙ СИНТЕЗ (HEAD) ---

class HybridColorHead(nn.Module):
    """Матричная трансформация + остаточная коррекция для высокого PSNR."""
    def __init__(self, in_channels=64):
        super().__init__()
        self.matrix_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, 12, kernel_size=1),
        )
        self.residual_gate = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat, src, residual):
        params = self.matrix_head(feat)
        b, _, h, w = params.shape
        m_delta = torch.tanh(params[:, :9]).view(b, 3, 3, h, w)
        bias = torch.tanh(params[:, 9:12]) * 0.2
        eye = torch.eye(3, device=params.device).view(1, 3, 3, 1, 1)
        matrix = eye + 0.1 * m_delta
        explicit = torch.einsum("boihw,bihw->bohw", matrix, src) + bias
        g = self.residual_gate(feat)
        return (explicit + g * residual).clamp(0.0, 1.0)

# --- 4. ОСНОВНОЙ СЛОЙ HGSA ---

class HyperGaussianPPA_Layer(nn.Module):
    """Слой, объединяющий Гауссовы поля и Pentamodal Attention."""
    def __init__(self, dim, M=16, num_concepts=5):
        super().__init__()
        self.M = M
        self.num_concepts = num_concepts
        self.v_parameters = nn.Parameter(torch.randn(1, num_concepts, dim, 1, 1))
        
        self.hyper_enc = HypernetworkParameterEncoder(M * 3, dim, dim, dim // 2)
        self.key_gen = nn.Conv2d(dim, num_concepts, kernel_size=1)
        self.main_path = GatedConvBlock(dim)

    def forward(self, x, global_code):
        b, c, h, w = x.shape
        params = self.hyper_enc(x, global_code).view(b, self.M, 3, h, w)
        
        a = torch.tanh(params[:, :, 0, :, :])
        mu = torch.sigmoid(params[:, :, 1, :, :])
        sigma = torch.sigmoid(params[:, :, 2, :, :]) * 0.5 + 0.1
        
        x_norm = torch.mean(torch.sigmoid(x), dim=1, keepdim=True)
        diff = (x_norm - mu) ** 2
        gaussians = a * torch.exp(-diff / (2 * sigma**2))
        
        keys = torch.softmax(self.key_gen(x), dim=1)
        combined_v = torch.sum(keys.unsqueeze(2) * self.v_parameters, dim=1)
        
        gaussian_weight = torch.sum(gaussians, dim=1, keepdim=True)
        modulation = combined_v * gaussian_weight
        
        q = self.main_path(x)
        return x + q * (1 + modulation)

# --- 5. ФИНАЛЬНАЯ АРХИТЕКТУРА ---

class HGSA_v1(nn.Module):
    def __init__(self, width=64, depth=6, M=16):
        super().__init__()
        self.coord_adder = AddCoords()
        self.stem = nn.Conv2d(3 + 2, width, kernel_size=3, padding=1)
        self.global_encoder = GlobalColorEncoder(channels=width, hidden=width)
        
        self.layers = nn.ModuleList([
            HyperGaussianPPA_Layer(width, M=M) for _ in range(depth)
        ])
        
        self.color_head = HybridColorHead(in_channels=width)
        self.final_res = nn.Conv2d(width, 3, 1)

    def forward(self, src):
        # src: [B, 3, H, W]
        x_in = self.coord_adder(src)
        feat = self.stem(x_in)
        
        global_code = self.global_encoder(feat)
        
        for layer in self.layers:
            feat = layer(feat, global_code)
            
        res = self.final_res(feat)
        return self.color_head(feat, src, res)

# Пример инициализации:
# model = HGSA_Net(width=64, depth=8, M=16)