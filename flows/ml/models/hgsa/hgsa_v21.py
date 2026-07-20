import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ─── 1. Вспомогательные модули ────────────────────────────────────────

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FiLM(nn.Module):
    """Feature-wise Linear Modulation для глобального кондиционирования."""
    def __init__(self, giv_dim, feat_dim):
        super().__init__()
        self.proj = nn.Linear(giv_dim, feat_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat, giv):
        gb = self.proj(giv).view(giv.size(0), -1, 1, 1)
        gamma, beta = gb.chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta

# ─── 2. Кодирование и извлечение признаков ───────────────────────────

class DegradationAwareConditioner(nn.Module):
    """Извлекает вектор состояния деградации g."""
    def __init__(self, in_c, giv_dim):
        super().__init__()
        stat_dim = in_c * 2 * (1 + 16 + 64)
        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, giv_dim * 2),
            GELU(),
            nn.Linear(giv_dim * 2, giv_dim),
            nn.LayerNorm(giv_dim)
        )

    def _scale_stats(self, x, size):
        mu = F.adaptive_avg_pool2d(x, size).flatten(1)
        mu2 = F.adaptive_avg_pool2d(x ** 2, size).flatten(1)
        var = (mu2 - mu ** 2).clamp(min=0)
        return torch.cat([mu, var], dim=1)

    def forward(self, x):
        s1 = self._scale_stats(x, 1)
        s4 = self._scale_stats(x, 4)
        s8 = self._scale_stats(x, 8)
        return self.mlp(torch.cat([s1, s4, s8], dim=1))

class FullResEncoder(nn.Module):
    """Энкодер для получения локального контекста f_ij."""
    def __init__(self, in_c, feat_dim, giv_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, feat_dim, 3, padding=1),
            nn.GroupNorm(4, feat_dim), GELU()
        )
        self.d1 = nn.Conv2d(feat_dim, feat_dim, 3, padding=1, dilation=1, groups=feat_dim)
        self.d2 = nn.Conv2d(feat_dim, feat_dim, 3, padding=2, dilation=2, groups=feat_dim)
        self.d4 = nn.Conv2d(feat_dim, feat_dim, 3, padding=4, dilation=4, groups=feat_dim)
        
        self.fuse = nn.Sequential(nn.Conv2d(feat_dim * 3, feat_dim, 1), nn.GroupNorm(4, feat_dim))
        self.film = FiLM(giv_dim, feat_dim)
        
        self.illu_conv = nn.Sequential(
            nn.Conv2d(in_c + 1, 16, 1),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, giv):
        illu_map = self.illu_conv(torch.cat([x, x.mean(1, keepdim=True)], dim=1))
        f = self.stem(x)
        feat = self.fuse(torch.cat([self.d1(f), self.d2(f), self.d4(f)], dim=1))
        return self.film(feat, giv), illu_map

# ─── 3. Ядро USGS + EOT Синтез ───────────────────────────────────────

class SAGFLayer(nn.Module):
    """
    Внутренняя сумма USGS: K_q(x) = sum_{m} a_m * exp(-||x - mu||^2 / 2sigma^2).
    Реализует адаптивное ядро Гиббса.
    """
    def __init__(self, feat_dim, Q, M):
        super().__init__()
        self.Q, self.M = Q, M
        self.params_per_m = 5 # a(1), mu(3), sigma(1)
        self.hyper = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            nn.Conv2d(feat_dim, Q * M * self.params_per_m, 1)
        )

    def forward(self, x, feat):
        B, C, H, W = x.shape
        p = self.hyper(feat).view(B, self.Q, self.M, self.params_per_m, H, W)
        
        # Параметры внутренних гауссиан
        a = torch.softmax(p[:, :, :, 0:1], dim=2) # Нормировка весов внутри ядра
        mu = torch.sigmoid(p[:, :, :, 1:4])
        sigma = F.softplus(p[:, :, :, 4:5]) + 1e-4
        
        # Вычисление расстояния от известного x_ij до mu_m
        x_exp = x.unsqueeze(1).unsqueeze(2) # [B, 1, 1, 3, H, W]
        dist_sq = torch.sum((x_exp - mu)**2, dim=3, keepdim=True)
        
        # Суммирование по M (внутренняя суперпозиция)
        kernels = a * torch.exp(-0.5 * dist_sq / (sigma**2))
        return torch.sum(kernels, dim=2) # [B, Q, 1, H, W]

class TransportOperatorHead(nn.Module):
    """
    Предсказывает параметры локальных отображений T_q(x).
    T_q(x) = C + A(x - mu) + B(x - mu)^2
    """
    def __init__(self, feat_dim, Q):
        super().__init__()
        self.Q = Q
        # Параметры: A(9) + B(9) + C(3) + mu(3) = 24
        self.params_per_q = 24
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            nn.Conv2d(feat_dim, Q * self.params_per_q, 1)
        )

    def forward(self, feat):
        B, _, H, W = feat.shape
        p = self.net(feat).view(B, self.Q, self.params_per_q, H, W)
        return p

class EOT_USGS_PixelWise_Block(nn.Module):
    """
    Синтез USGS и EOT с попиксельной нормировкой плана и барицентрической проекцией.
    """
    def __init__(self, in_c=3, Q=12, M=4, feat_dim=64, giv_dim=64):
        super().__init__()
        self.Q, self.M = Q, M
        self.gibbs_layer = SAGFLayer(feat_dim, Q, M)
        self.transport_head = TransportOperatorHead(feat_dim, Q)
        
        # Потенциал Шрёдингера psi_q (внешняя суперпозиция)
        self.phi_net = nn.Sequential(
            nn.Conv2d(feat_dim, Q, 1),
            nn.Softplus() 
        )

    def forward(self, x_img, feat, giv):
        B, C, H, W = x_img.shape
        
        # 1. Вычисляем локальные ядра Гиббса K_q(x)
        K = self.gibbs_layer(x_img, feat) # [B, Q, 1, H, W]
        
        # 2. Вычисляем потенциалы psi_q и формируем нормированный план pi
        psi = self.phi_net(feat).unsqueeze(2) # [B, Q, 1, H, W]
        
        # Попиксельная нормировка (учитываем, что x_ij известен)
        plan_raw = K * psi
        pi_q = plan_raw / (torch.sum(plan_raw, dim=1, keepdim=True) + 1e-8)
        
        # 3. Вычисляем локальные транспорты T_q(x)
        p = self.transport_head(feat)
        A_raw = p[:, :, 0:9].view(B, self.Q, 3, 3, H, W)
        B_raw = p[:, :, 9:18].view(B, self.Q, 3, 3, H, W)
        C_loc = torch.sigmoid(p[:, :, 18:21])
        mu_ref = torch.sigmoid(p[:, :, 21:24])
        
        # Формируем операторы (A = I + delta)
        eye = torch.eye(3, device=x_img.device).view(1, 1, 3, 3, 1, 1)
        A = eye + 0.1 * torch.tanh(A_raw)
        B = 0.02 * torch.tanh(B_raw)
        
        # Применяем T_q к x_ij
        diff = (x_img.unsqueeze(1) - mu_ref).permute(0, 1, 3, 4, 2).unsqueeze(-1)
        A_mat = A.permute(0, 1, 4, 5, 2, 3) 
        B_mat = B.permute(0, 1, 4, 5, 2, 3)
        
        # T_q(x) = C + A(x-mu) + B(x-mu)^2
        trans = torch.matmul(A_mat, diff) + torch.matmul(B_mat, diff**2)
        T_q = C_loc + trans.squeeze(-1).permute(0, 1, 4, 2, 3)
        
        # 4. Барицентрическая проекция
        return torch.sum(pi_q * T_q, dim=1)

# ─── 4. Финальная сборка и уточнение ─────────────────────────────────

class LaplacianGatedFusion(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.lap = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c)
        self.gate = nn.Sequential(
            nn.Conv2d(in_c + out_c + 1, 16, 3, padding=1),
            GELU(),
            nn.Conv2d(16, out_c, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_c + out_c, out_c * 2, 1),
            nn.GroupNorm(2, out_c * 2), # 6 каналов делятся на 2 группы
            GELU(),
            nn.Conv2d(out_c * 2, out_c, 3, padding=1)
        )

    def forward(self, x_orig, transported, illu_map):
        x_detail = x_orig - self.lap(x_orig)
        g = self.gate(torch.cat([x_orig, transported, illu_map], dim=1))
        blended = transported * g + (x_orig + x_detail) * (1 - g)
        return self.refine(torch.cat([blended, x_orig], dim=1))

class HGSA_USGS_EOT_v21(nn.Module):
    """
    HGSA v21: Полный синтез USGS и EOT.
    Реализует попиксельный транспортный план и барицентрическую проекцию.
    """
    def __init__(self, in_channels=3, out_channels=3, Q=12, M=4):
        super().__init__()
        GIV_DIM, FEAT_DIM = 64, 64
        
        self.conditioner = DegradationAwareConditioner(in_channels, GIV_DIM)
        self.encoder = FullResEncoder(in_channels, FEAT_DIM, GIV_DIM)
        self.eot_usgs = EOT_USGS_PixelWise_Block(in_channels, Q, M, FEAT_DIM, GIV_DIM)
        self.fusion = LaplacianGatedFusion(in_channels, out_channels)

    def forward(self, src):
        # 1. Извлечение глобального и локального контекста
        giv = self.conditioner(src)
        feat, illu_map = self.encoder(src, giv)
        
        # 2. Попиксельный синтез USGS + EOT
        # Здесь вычисляется план pi(x,y) и барицентрическая проекция T(x)
        transported_x = self.eot_usgs(src, feat, giv)
        
        # 3. Финальное уточнение деталей
        final_out = self.fusion(src, transported_x, illu_map)
        
        if self.training:
            return {'res': (final_out, transported_x)}
        return {'res': final_out}