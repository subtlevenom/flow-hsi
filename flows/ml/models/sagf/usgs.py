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

# --- ВНИМАНИЕ (ATTENTION) МОДУЛИ ---

class ChannelAttention(nn.Module):
    """Squeeze-Excitation block: learn relative importance of channels."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 1), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class SpatialAttention(nn.Module):
    """Learn spatial importance map (which pixels matter most)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_map = self.sigmoid(self.conv(avg_pool))
        return x * spatial_map

class CrossScaleAttention(nn.Module):
    """Multi-head attention across pooling scales."""
    def __init__(self, hidden, num_scales=4, num_heads=4):
        super().__init__()
        self.hidden = hidden
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        assert hidden % num_heads == 0, "hidden must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(hidden, 3 * hidden)
        self.scale_proj = nn.Linear(3 * hidden * num_scales, hidden * 2)
        
    def forward(self, scale_features):
        B = scale_features[0].size(0)
        stacked = torch.cat([f.view(B, -1) for f in scale_features], dim=1)
        qkv = self.qkv_proj(stacked.view(B * self.num_scales, self.hidden)).view(B, self.num_scales, -1)
        Q, K, V = qkv.chunk(3, dim=-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, V)
        attn_out_flat = attn_out.reshape(B, -1)
        out = self.scale_proj(attn_out_flat)
        return out

# --- УЛУЧШЕННЫЙ ЭНКОДЕР ПАРАМЕТРОВ ---

class HypernetworkParameterEncoder(nn.Module):
    """Compact cmKAN-like hyper-encoder for Gaussian basis parameters.

    Uses local depthwise-separable convs plus a shared global latent code to
    modulate spatial features before predicting basis parameters.
    """

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
        gamma = 1.0 + 0.5 * gamma
        f = gamma * f + beta
        return self.out_head(f)


class GlobalColorEncoder(nn.Module):
    """Extract global color statistics for source-only feature learning.
    
    Uses multi-scale pooling with cross-scale attention (Phase 2 improvement).
    Learns which scales to emphasize for better color distribution modeling.
    """
    def __init__(self, channels=64, hidden=64, pool_scales=(1, 2, 4, 8), num_heads=4):
        super().__init__()
        self.pool_scales = tuple(pool_scales)
        self.hidden = hidden
        attn_heads = num_heads
        while attn_heads > 1 and (hidden % attn_heads != 0):
            attn_heads -= 1
        self.feat_x = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Token attention across pooled scales (cmKAN-style compact latent mixing).
        self.token_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=attn_heads, batch_first=True)
        self.token_gate = nn.Linear(hidden, 1)

        # Project attention-weighted token descriptor to color code.
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )

    @staticmethod
    def _pooled_token(feat, size):
        pooled = F.adaptive_avg_pool2d(feat, size)
        return pooled.mean(dim=(2, 3))

    def forward(self, feat):
        f_x = self.feat_x(feat)

        # Build one token per pooling scale.
        tokens = []
        for s in self.pool_scales:
            tokens.append(self._pooled_token(f_x, s))
        token_seq = torch.stack(tokens, dim=1)

        token_ctx, _ = self.token_attn(token_seq, token_seq, token_seq)
        token_w = F.softmax(self.token_gate(token_ctx).squeeze(-1), dim=1)
        z = (token_ctx * token_w.unsqueeze(-1)).sum(dim=1)

        # Project to global latent code.
        color_code = self.proj(z)
        return color_code.unsqueeze(-1).unsqueeze(-1)


class StyleFiLM(nn.Module):
    """Inject global style via per-channel affine modulation."""

    def __init__(self, channels=64, style_channels=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(style_channels, channels * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
        )

    def forward(self, feat, style):
        gamma, beta = torch.chunk(self.proj(style), 2, dim=1)
        gamma = torch.sigmoid(gamma) * 2.0
        beta = torch.tanh(beta)
        return gamma * feat + beta


class HybridColorHead(nn.Module):
    """Explicit local 3x3 transform plus bounded residual branch."""

    def __init__(self, in_channels=64):
        super().__init__()
        hidden = max(in_channels, 32)
        self.matrix_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 12, kernel_size=1),
        )
        self.residual_gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _explicit_transform(self, feat, src):
        params = self.matrix_head(feat)
        b, _, h, w = params.shape
        m_delta = torch.tanh(params[:, :9]).view(b, 3, 3, h, w)
        bias = torch.tanh(params[:, 9:12]) * 0.2
        eye = torch.eye(3, device=params.device, dtype=params.dtype).view(1, 3, 3, 1, 1)
        matrix = eye + 0.1 * m_delta
        return torch.einsum("boihw,bihw->bohw", matrix, src) + bias

    def forward(self, feat, src, residual):
        explicit = self._explicit_transform(feat, src)
        g = self.residual_gate(feat)
        out = explicit + g * residual
        return out.clamp(0.0, 1.0)


class NonLinearAdjust(nn.Module):
    """Hybrid nonlinear image adjustment with identity-safe residual correction."""

    def __init__(self, feat_channels=128, illum_channels=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 + feat_channels + illum_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 3, kernel_size=3, padding=1),
        )
        # Start with a small residual contribution for stable early training.
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_base, feat, illum_code):
        if illum_code.size(-2) == 1 and illum_code.size(-1) == 1:
            illum_code = illum_code.expand(-1, -1, x_base.size(-2), x_base.size(-1))
        delta = torch.tanh(self.net(torch.cat([x_base, feat, illum_code], dim=1)))
        out = x_base + self.alpha * delta
        return out.clamp(0.0, 1.0)

# --- ОСНОВНАЯ МОДЕЛЬ HC-USGS ---

class USGS(nn.Module):
    def __init__(self, M=32, Q=16, hidden=128):
        super().__init__()
        self.M = M  # Количество Гауссиан
        self.Q = Q  # Количество адаптивных узлов сканирования
        self.hidden = hidden

        self.coord_adder = AddCoords()
        
        # Stem: принимает RGB + 2 канала координат
        self.stem = nn.Sequential(
            nn.Conv2d(3 + 2, hidden, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden),
            nn.SiLU(),
        )
        
        # Global color encoder (reference-free style learning)
        self.color_encoder = GlobalColorEncoder(channels=hidden, hidden=hidden)
        
        # Hypernetwork-based parameter encoders (spatially-varying)
        enc_hidden = max(hidden // 2, 32)
        self.enc_amplitude = HypernetworkParameterEncoder(
            out_dim=M,
            in_channels=hidden,
            global_channels=hidden,
            hidden=enc_hidden,
        )
        self.enc_center = HypernetworkParameterEncoder(
            out_dim=M,
            in_channels=hidden,
            global_channels=hidden,
            hidden=enc_hidden,
        )
        self.enc_sigma = HypernetworkParameterEncoder(
            out_dim=M,
            in_channels=hidden,
            global_channels=hidden,
            hidden=enc_hidden,
        )

        # Генератор адаптивной сетки узлов
        self.node_generator = nn.Sequential(
            ResidualContextBlock(hidden),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, Q),
            nn.Sigmoid()
        )

        # Блок внешней суперпозиции (chi)
        self.chi_superposition = nn.Sequential(
            nn.Conv2d(Q * 3, Q * 6, kernel_size=3, padding=1, groups=Q),
            nn.SiLU(),
            nn.Conv2d(Q * 6, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 3, kernel_size=1)
        )

        self.psi_norm = nn.InstanceNorm2d(Q * 3)
        self.color_head = HybridColorHead(in_channels=hidden)
        illum_channels = max(hidden // 8, 16)
        self.illum_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, illum_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(illum_channels, illum_channels, kernel_size=1),
            nn.Tanh(),
        )
        self.non_linear_adjust = NonLinearAdjust(
            feat_channels=hidden,
            illum_channels=illum_channels,
            hidden=max(hidden // 2, 32),
        )

    def forward(self, src, return_aux=False):
        """Forward pass (completely reference-free).
        
        Args:
            src: source image [B, 3, H, W]
            return_aux: whether to return auxiliary outputs dict
            
        Returns:
            if return_aux=True: (output [B, 3, H, W], aux_dict)
            else: output [B, 3, H, W]
        """
        if src.dim() == 3:
            src = src.unsqueeze(0)

        x = src
        b, _, h, w = x.shape
        
        # 1. Extract features with coordinates
        x_coords = self.coord_adder(x)
        feat = self.stem(x_coords)

        # 2. Learn global color distribution from source (reference-free)
        color_code = self.color_encoder(feat)  # [B, hidden, 1, 1]
        
        # 3. Generate spatially-varying Gaussian parameters (hypernetwork-inspired)
        a = torch.tanh(self.enc_amplitude(feat, color_code)).unsqueeze(1)
        mu = torch.sigmoid(self.enc_center(feat, color_code)).unsqueeze(1)
        sigma = (torch.sigmoid(self.enc_sigma(feat, color_code)) * 0.45 + 0.05).unsqueeze(1)

        # 4. Adaptive scan nodes (reference-free)
        q_nodes = self.node_generator(feat)
        q_nodes = torch.sigmoid(q_nodes).view(b, 1, 1, self.Q, 1, 1)
        q_nodes, _ = torch.sort(q_nodes, dim=3)

        # 5. Nonlinear adjustment conditioned on global illumination code
        illum_code = self.illum_global(feat)
        x_adj = self.non_linear_adjust(x, feat, illum_code)

        # 6. Gaussian superposition (reference-free basis function KAN-like)
        diff = (q_nodes - mu.unsqueeze(3)) ** 2
        gaussians = a.unsqueeze(3) * torch.exp(-diff / (2 * sigma.unsqueeze(3)**2))
        S_M_q = torch.sum(gaussians, dim=2)

        # 7. External superposition (chi)
        chi_input = (S_M_q * x_adj.unsqueeze(2)).view(b, self.Q * 3, h, w)
        chi_input = self.psi_norm(chi_input)
        color_residual = 0.25 * torch.tanh(self.chi_superposition(chi_input))

        # 8. Final color output
        out = self.color_head(feat, x_adj, color_residual)

        # Return learnable parameters for regularization
        aux = {
            "amplitude": a.squeeze(1),  # [B, M, H, W]
            "center": mu.squeeze(1),    # [B, M, H, W]
            "sigma": sigma.squeeze(1),  # [B, M, H, W]
        }

        if return_aux:
            return out, aux
        return out