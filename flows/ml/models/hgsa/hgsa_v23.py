import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 1. БАЗОВЫЕ УТИЛИТЫ И КОНДИЦИОНИРОВАНИЕ ──────────────────────────

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FiLM(nn.Module):
    """Feature-wise Linear Modulation для глобального контекста."""
    def __init__(self, giv_dim, feat_dim):
        super().__init__()
        self.proj = nn.Linear(giv_dim, feat_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat, giv):
        gb = self.proj(giv).view(giv.size(0), -1, 1, 1)
        gamma, beta = gb.chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta

class DegradationAwareConditioner(nn.Module):
    """Извлекает вектор состояния деградации g из глобальной статистики."""
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

# ─── 2. МНОГОМАСШТАБНЫЙ ЭНКОДЕР (v23) ────────────────────────────────

class MSResBlock(nn.Module):
    """Лёгкий остаточный блок (depthwise + pointwise) для одного масштаба."""
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.GroupNorm(4, dim)
        self.act = GELU()
        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return x + self.pw(self.act(self.norm(self.dw(x))))

class MultiScaleEncoder(nn.Module):
    """Многомасштабный (пирамидальный) энкодер с оценкой карты освещенности.

    В отличие от одноуровневого dilated-энкодера v22, признаки извлекаются
    на нескольких разрешениях (пирамида /1, /2, /4, ...) и агрегируются обратно
    к полному разрешению в U-Net-подобном декодере. Это увеличивает
    эффективное рецептивное поле и ёмкость представления при сохранении
    полноразмерного выхода признаков.
    """
    def __init__(self, in_c, feat_dim, giv_dim, scales=3):
        super().__init__()
        assert scales >= 1
        self.scales = scales

        self.stem = nn.Sequential(
            nn.Conv2d(in_c, feat_dim, 3, padding=1),
            nn.GroupNorm(4, feat_dim), GELU()
        )

        # Блоки обработки на каждом масштабе пирамиды.
        self.blocks = nn.ModuleList([MSResBlock(feat_dim) for _ in range(scales)])
        # Понижение разрешения между масштабами (strided conv).
        self.downs = nn.ModuleList([
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1)
            for _ in range(scales - 1)
        ])
        # Слияние в декодере (skip + апсемпл верхнего уровня).
        self.reduce = nn.ModuleList([
            nn.Sequential(nn.Conv2d(feat_dim * 2, feat_dim, 1), GELU())
            for _ in range(scales - 1)
        ])

        # Локальный dilated-контекст полного разрешения (детали, как в v22).
        self.d1 = nn.Conv2d(feat_dim, feat_dim, 3, padding=1, dilation=1, groups=feat_dim)
        self.d2 = nn.Conv2d(feat_dim, feat_dim, 3, padding=2, dilation=2, groups=feat_dim)
        self.d4 = nn.Conv2d(feat_dim, feat_dim, 3, padding=4, dilation=4, groups=feat_dim)

        # Финальное слияние многомасштабного и локального контекста.
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 4, feat_dim, 1), nn.GroupNorm(4, feat_dim)
        )
        self.film = FiLM(giv_dim, feat_dim)

        self.illu_conv = nn.Sequential(
            nn.Conv2d(in_c + 1, 16, 1),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, giv):
        illu_map = self.illu_conv(torch.cat([x, x.mean(1, keepdim=True)], dim=1))

        s = self.stem(x)

        # Кодировщик: строим пирамиду признаков.
        feats = [self.blocks[0](s)]
        for i in range(self.scales - 1):
            down = self.downs[i](feats[-1])
            feats.append(self.blocks[i + 1](down))

        # Декодер: агрегируем сверху вниз к полному разрешению.
        top = feats[-1]
        for i in range(self.scales - 2, -1, -1):
            up = F.interpolate(top, size=feats[i].shape[-2:], mode='bilinear', align_corners=False)
            top = self.reduce[i](torch.cat([up, feats[i]], dim=1))
        ms_feat = top  # полное разрешение, feat_dim

        # Локальный контекст на признаках полного разрешения.
        local = torch.cat([self.d1(ms_feat), self.d2(ms_feat), self.d4(ms_feat)], dim=1)
        feat = self.fuse(torch.cat([ms_feat, local], dim=1))
        return self.film(feat, giv), illu_map

# ─── 3. ГИПЕР-KAN И ТРАНСПОРТНЫЕ МОДУЛИ ──────────────────────────────

class HyperKAN_Cell(nn.Module):
    """Ячейка гиперсети с нелинейным KAN-базисом."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.base = nn.Conv2d(in_dim, out_dim, 1)
        self.spline = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, groups=in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 1)
        )
        self.combine = nn.Parameter(torch.ones(1, out_dim, 1, 1) * 0.5)

    def forward(self, x):
        return self.base(x) + self.combine * torch.tanh(self.spline(x))

class SAGFLayer_KAN(nn.Module):
    """Адаптивное ядро Гиббса (USGS) с KAN-гиперсетью."""
    def __init__(self, feat_dim, Q, M):
        super().__init__()
        self.Q, self.M = Q, M
        self.hyper = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            HyperKAN_Cell(feat_dim, Q * M * 5)
        )

    def forward(self, x, feat):
        B, C, H, W = x.shape
        p = self.hyper(feat).view(B, self.Q, self.M, 5, H, W)
        a = torch.softmax(p[:, :, :, 0:1], dim=2)
        mu = torch.sigmoid(p[:, :, :, 1:4])
        sigma = F.softplus(p[:, :, :, 4:5]) + 1e-4

        x_exp = x.unsqueeze(1).unsqueeze(2)
        dist_sq = torch.sum((x_exp - mu)**2, dim=3, keepdim=True)
        kernels = a * torch.exp(-0.5 * dist_sq / (sigma**2))
        return torch.sum(kernels, dim=2)

class TransportHead_KAN(nn.Module):
    """Генератор локальных транспортных операторов Tq."""
    def __init__(self, feat_dim, Q):
        super().__init__()
        self.Q = Q
        self.hyper = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            HyperKAN_Cell(feat_dim, Q * 24)
        )

    def forward(self, feat):
        B, _, H, W = feat.shape
        return self.hyper(feat).view(B, self.Q, 24, H, W)

class EOT_USGS_Volga_Block(nn.Module):
    """Ядро синтеза EOT + USGS с учетом освещенности."""
    def __init__(self, in_c=3, Q=16, M=4, feat_dim=64):
        super().__init__()
        self.Q, self.M = Q, M
        self.gibbs_layer = SAGFLayer_KAN(feat_dim, Q, M)
        self.transport_head = TransportHead_KAN(feat_dim, Q)
        self.phi_net = nn.Sequential(
            HyperKAN_Cell(feat_dim + 1, Q),
            nn.Softplus()
        )

    def forward(self, x_img, feat, illu_map):
        B, C, H, W = x_img.shape
        K = self.gibbs_layer(x_img, feat)
        psi = self.phi_net(torch.cat([feat, illu_map], dim=1)).unsqueeze(2)
        pi_q = (K * psi) / (torch.sum(K * psi, dim=1, keepdim=True) + 1e-8)

        p = self.transport_head(feat)
        A_raw = p[:, :, 0:9].view(B, self.Q, 3, 3, H, W)
        B_raw = p[:, :, 9:18].view(B, self.Q, 3, 3, H, W)
        C_loc = torch.sigmoid(p[:, :, 18:21])
        mu_ref = torch.sigmoid(p[:, :, 21:24])

        eye = torch.eye(3, device=x_img.device).view(1, 1, 3, 3, 1, 1)
        A = eye + 0.3 * torch.tanh(A_raw)
        B = 0.05 * torch.tanh(B_raw)

        diff = (x_img.unsqueeze(1) - mu_ref).permute(0, 1, 3, 4, 2).unsqueeze(-1)
        A_mat = A.permute(0, 1, 4, 5, 2, 3)
        B_mat = B.permute(0, 1, 4, 5, 2, 3)

        trans = torch.matmul(A_mat, diff) + torch.matmul(B_mat, diff**2)
        T_q = C_loc + trans.squeeze(-1).permute(0, 1, 4, 2, 3)

        return torch.sum(pi_q * T_q, dim=1)

# ─── 4. ФИНАЛЬНАЯ СБОРКА И УТОЧНЕНИЕ ─────────────────────────────────

class LaplacianGatedFusion(nn.Module):
    """Слияние с сохранением деталей через Лапласиан."""
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
            nn.GroupNorm(2, out_c * 2),
            GELU(),
            nn.Conv2d(out_c * 2, out_c, 3, padding=1)
        )

    def forward(self, x_orig, transported, illu_map):
        x_detail = x_orig - self.lap(x_orig)
        g = self.gate(torch.cat([x_orig, transported, illu_map], dim=1))
        blended = transported * g + (x_orig + x_detail) * (1 - g)
        return self.refine(torch.cat([blended, x_orig], dim=1))

class HGSA_USGS_EOT_v23(nn.Module):
    """
    Версия v23: увеличенная ёмкость + многомасштабный (пирамидальный) энкодер.

    Расширяет v22 двумя элементами плана:
      #4  — увеличение ёмкости (feat_dim/giv_dim, число транспортных мод Q)
            и многомасштабный энкодер (MultiScaleEncoder) вместо
            одноуровневого dilated-энкодера;
      #13 — поддержка переноса обучения (pretrain → fine-tune) реализована
            на стороне пайплайна (загрузка предобученных весов, поэтапная
            заморозка, пониженный LR дообучения).

    Ёмкость (feat_dim, giv_dim, Q, M, число масштабов) конфигурируема, что
    позволяет запускать как «лёгкую» предобучающую, так и «полную» версии
    из одной кодовой базы.
    """
    def __init__(self, in_channels=3, out_channels=3, Q=24, M=4,
                 feat_dim=96, giv_dim=96, scales=3):
        super().__init__()
        FD, GD = feat_dim, giv_dim

        self.conditioner = DegradationAwareConditioner(in_channels, GD)
        self.encoder = MultiScaleEncoder(in_channels, FD, GD, scales=scales)
        self.eot_usgs = EOT_USGS_Volga_Block(in_channels, Q, M, FD)
        self.fusion = LaplacianGatedFusion(in_channels, out_channels)

    def forward(self, src):
        # 1. Глобальный и локальный контекст
        giv = self.conditioner(src)
        feat, illu_map = self.encoder(src, giv)

        # 2. Транспортный синтез (Hyper-KAN + EOT)
        transported_x = self.eot_usgs(src, feat, illu_map)

        # 3. Финальное уточнение
        final_out = self.fusion(src, transported_x, illu_map)

        if self.training:
            return {'res': (final_out, transported_x)}
        return {'res': final_out}
