import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossOscillatorAttention(nn.Module):
    """Cross-attention where Q/K/V are oscillator-evolved features.

    Adds a resonance compatibility term to logits so positions with similar
    oscillator dynamics (omega, zeta) attend more strongly.
    """

    def __init__(
        self,
        channels: int,
        pool: int = 4,
        steps: int = 3,
        dt: float = 0.2,
        omega_max: float = 2.0,
        resonance_weight: float = 0.15,
    ):
        super().__init__()
        self.pool = pool
        self.steps = steps
        self.dt = dt
        self.omega_max = omega_max
        self.resonance_weight = resonance_weight

        self.q_in = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_in = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_in = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.q_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.q_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.k_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.v_pw = nn.Conv2d(channels, channels, 1, bias=False)

        self.omega_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.zeta_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.omega_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.zeta_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.omega_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.zeta_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.pool, stride=self.pool)

    def _osc_step(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        force: torch.Tensor,
        omega: torch.Tensor,
        zeta: torch.Tensor,
        dw: nn.Conv2d,
        pw: nn.Conv2d,
    ):
        coupling = dw(x) + pw(x)
        accel = force + coupling - 2.0 * zeta * omega * v - (omega**2) * x
        v = v + accel * self.dt
        x = x + v * self.dt
        return x, v

    def _evolve(
        self,
        feat: torch.Tensor,
        proj: nn.Conv2d,
        omega_head: nn.Conv2d,
        zeta_head: nn.Conv2d,
        dw: nn.Conv2d,
        pw: nn.Conv2d,
    ):
        force = proj(feat)
        omega = torch.sigmoid(omega_head(feat)) * self.omega_max
        zeta = torch.sigmoid(zeta_head(feat))
        x = torch.zeros_like(force)
        v = torch.zeros_like(force)
        for _ in range(self.steps):
            x, v = self._osc_step(x, v, force, omega, zeta, dw, pw)
        return x, omega, zeta

    def forward(self, src_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        s = self._pool(src_feat)
        r = self._pool(ref_feat)

        q_feat, q_w, q_z = self._evolve(s, self.q_in, self.omega_q, self.zeta_q, self.q_dw, self.q_pw)
        k_feat, k_w, k_z = self._evolve(r, self.k_in, self.omega_k, self.zeta_k, self.k_dw, self.k_pw)
        v_feat, _, _ = self._evolve(r, self.v_in, self.omega_v, self.zeta_v, self.v_dw, self.v_pw)

        q = q_feat.flatten(2).transpose(1, 2)
        k = k_feat.flatten(2).transpose(1, 2)
        v = v_feat.flatten(2).transpose(1, 2)

        qwf = q_w.flatten(2).transpose(1, 2)
        kwf = k_w.flatten(2).transpose(1, 2)
        qzf = q_z.flatten(2).transpose(1, 2)
        kzf = k_z.flatten(2).transpose(1, 2)

        scale = q.shape[-1] ** -0.5
        base_logits = torch.bmm(q, k.transpose(1, 2)) * scale

        # Resonance affinity: smaller parameter distance -> higher attention score.
        dist_w = torch.cdist(qwf, kwf, p=2) / (q.shape[-1] ** 0.5)
        dist_z = torch.cdist(qzf, kzf, p=2) / (q.shape[-1] ** 0.5)
        resonance = -(dist_w + dist_z)

        attn = torch.softmax(base_logits + self.resonance_weight * resonance, dim=-1)
        ctx = torch.bmm(attn, v).transpose(1, 2).view_as(s)
        ctx = self.out_proj(ctx)
        if ctx.shape[-2:] != src_feat.shape[-2:]:
            ctx = F.interpolate(ctx, size=src_feat.shape[-2:], mode="bilinear", align_corners=False)

        g = self.gate(torch.cat([src_feat, ctx], dim=1))
        return src_feat + g * ctx


class IlluminationEstimator(nn.Module):
    """Estimate illumination features and map from RGB + grayscale."""

    def __init__(self, channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.map_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        gray = x.mean(dim=1, keepdim=True)
        feat = self.stem(torch.cat([x, gray], dim=1))
        illum = torch.sigmoid(self.map_head(feat))
        return feat, illum


class OscillatorCell(nn.Module):
    """One symplectic-Euler step of a damped driven oscillator."""

    def __init__(self, channels: int):
        super().__init__()
        # Local spatial coupling (per channel)
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        # Cross-channel coupling for color interaction
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        force: torch.Tensor,
        omega: torch.Tensor,
        zeta: torch.Tensor,
        dt: float,
    ):
        coupling = self.dw(x) + self.pw(x)
        accel = force + coupling - 2.0 * zeta * omega * v - (omega**2) * x
        v_new = v + accel * dt
        x_new = x + v_new * dt
        return x_new, v_new


class RefStyleEncoder(nn.Module):
    """Multi-scale reference style encoder using mean/std statistics."""

    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 32,
        style_channels: int = 64,
        pool_sizes=(1, 2, 4),
    ):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        stat_dim = feat_channels * 2 * sum(s * s for s in self.pool_sizes)
        self.proj = nn.Sequential(
            nn.Linear(stat_dim, style_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_channels * 2, style_channels),
        )

    @staticmethod
    def _pool_mean_std(f: torch.Tensor, size: int):
        mean = F.adaptive_avg_pool2d(f, size)
        sq_mean = F.adaptive_avg_pool2d(f * f, size)
        var = (sq_mean - mean * mean).clamp(min=1e-6)
        std = torch.sqrt(var)
        return mean.flatten(1), std.flatten(1)

    def forward(self, ref: torch.Tensor) -> torch.Tensor:
        f = self.feat(ref)
        stats = []
        for s in self.pool_sizes:
            mu, sigma = self._pool_mean_std(f, s)
            stats.extend([mu, sigma])
        z = torch.cat(stats, dim=1)
        return self.proj(z).unsqueeze(-1).unsqueeze(-1)


class SourceStylePredictor(nn.Module):
    """Predict a target-style proxy from source features for reference-free inference.

    During training, this branch can be supervised using true reference style.
    During inference, it replaces missing ref input.
    """

    def __init__(self, channels: int, style_channels: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.style_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, style_channels, kernel_size=1),
        )

    def forward(self, src_feat: torch.Tensor, illum_feat: torch.Tensor):
        proxy_feat = self.feat(torch.cat([src_feat, illum_feat], dim=1))
        style = self.style_head(proxy_feat)
        return style, proxy_feat


class HyperEncoder(nn.Module):
    """Predicts oscillator and adaptive normalization parameters."""

    def __init__(self, channels: int, style_channels: int, omega_max: float = 2.0):
        super().__init__()
        self.omega_max = omega_max
        hidden = max(channels // 2, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels + style_channels, hidden, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 4 * channels, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, style: torch.Tensor):
        style_map = style.expand(feat.shape[0], style.shape[1], feat.shape[2], feat.shape[3])
        w, z, g, b = torch.chunk(self.net(torch.cat([feat, style_map], dim=1)), 4, dim=1)
        omega = torch.sigmoid(w) * self.omega_max
        zeta = torch.sigmoid(z)
        gamma = torch.sigmoid(g) * 2.0
        beta = torch.tanh(b)
        return omega, zeta, gamma, beta


class OscillatorLayer(nn.Module):
    """Unrolled oscillator plus style-conditioned adaptive normalization."""

    def __init__(
        self,
        channels: int,
        style_channels: int,
        steps: int = 10,
        dt: float = 0.3,
    ):
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.cell = OscillatorCell(channels)
        self.hyper = HyperEncoder(channels, style_channels)
        self.in_norm = nn.InstanceNorm2d(channels, affine=False)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-2)

    def forward(
        self,
        force: torch.Tensor,
        style: torch.Tensor,
        illum_feat: torch.Tensor,
        state=None,
    ):
        force = force + 0.2 * illum_feat
        omega, zeta, gamma, beta = self.hyper(force, style)

        if state is None:
            x = torch.zeros_like(force)
            v = torch.zeros_like(force)
        else:
            x, v = state

        for _ in range(self.steps):
            x, v = self.cell(x, v, force, omega, zeta, self.dt)

        out = gamma * self.in_norm(force + x) + beta
        out = out + self.layer_scale * self.refine(out)
        return out, (x, v)


class HybridColorHead(nn.Module):
    """Blend explicit linear color transform with nonlinear residual delta."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels * 2, 32)

        self.matrix_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 12, kernel_size=1),  # 9 matrix + 3 bias
        )

        self.spline_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _pixel_color_transform(self, feat: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        params = self.matrix_head(feat)
        b, _, h, w = params.shape

        m_delta = torch.tanh(params[:, :9]).view(b, 3, 3, h, w)
        bias = params[:, 9:12]

        eye = torch.eye(3, device=params.device, dtype=params.dtype).view(1, 3, 3, 1, 1)
        matrix = eye + 0.1 * m_delta

        out = torch.einsum("boihw,bihw->bohw", matrix, src) + bias
        return out

    def forward(self, feat: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        explicit = self._pixel_color_transform(feat, src)
        delta = 0.25 * torch.tanh(self.spline_head(feat))
        g = self.gate(feat)
        out = explicit + g * delta
        return out.clamp(0.0, 1.0)


class ResonantColorMatcher(nn.Module):
    """Reference-based image color matcher with resonant oscillator stack."""

    def __init__(
        self,
        channels: int = 32,
        style_channels: int = 64,
        num_layers: int = 6,
        steps: int = 10,
        dt: float = 0.25,
    ):
        super().__init__()

        self.illum = IlluminationEstimator(channels)
        self.src_shortcut = nn.Conv2d(3, channels, kernel_size=1)
        self.src_body = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.ref_body = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.ref_enc = RefStyleEncoder(
            in_channels=3,
            feat_channels=channels,
            style_channels=style_channels,
        )
        self.src_style_pred = SourceStylePredictor(
            channels=channels,
            style_channels=style_channels,
        )

        self.osc_layers = nn.ModuleList(
            [
                OscillatorLayer(
                    channels=channels,
                    style_channels=style_channels,
                    steps=steps,
                    dt=dt,
                )
                for _ in range(num_layers)
            ]
        )
        self.cross_attn_layers = nn.ModuleList(
            [CrossOscillatorAttention(channels=channels, pool=4) for _ in range(num_layers)]
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = HybridColorHead(channels)

    def forward(
        self,
        src: torch.Tensor,
        ref: torch.Tensor = None,
        use_ref_when_available: bool = True,
        return_aux: bool = False,
    ):
        if src.dim() == 3:
            src = src.unsqueeze(0)
        if ref is not None and ref.dim() == 3:
            ref = ref.unsqueeze(0)

        illum_feat, illum_map = self.illum(src)
        feat = self.src_body(src) + self.src_shortcut(src)
        feat = self.fusion(torch.cat([feat, illum_feat], dim=1))

        pred_style, proxy_ref_feat = self.src_style_pred(feat, illum_feat)

        if ref is not None and use_ref_when_available:
            ref_feat = self.ref_body(ref)
            ref_style = self.ref_enc(ref)
            # Blend true reference style with source-predicted style to reduce
            # train/inference mismatch.
            style = 0.7 * ref_style + 0.3 * pred_style
            aux = {
                "pred_style": pred_style,
                "target_style": ref_style,
                "proxy_ref_feat": proxy_ref_feat,
                "target_ref_feat": ref_feat,
            }
        else:
            ref_feat = proxy_ref_feat
            style = pred_style
            aux = {
                "pred_style": pred_style,
                "target_style": None,
                "proxy_ref_feat": proxy_ref_feat,
                "target_ref_feat": None,
            }

        state = None
        for xattn, layer in zip(self.cross_attn_layers, self.osc_layers):
            feat = xattn(feat, ref_feat)
            feat, state = layer(feat, style, illum_feat * illum_map, state=state)

        out = self.head(feat, src)
        if return_aux:
            return out, aux
        return out


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)).pow(2.4))


def _rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    rgb_lin = _srgb_to_linear(rgb.clamp(0.0, 1.0))
    r = rgb_lin[:, 0:1]
    g = rgb_lin[:, 1:2]
    b = rgb_lin[:, 2:3]

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return torch.cat([x, y, z], dim=1)


def _xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x = xyz[:, 0:1] / xn
    y = xyz[:, 1:2] / yn
    z = xyz[:, 2:3] / zn

    eps = 216.0 / 24389.0
    k = 24389.0 / 27.0

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, t.pow(1.0 / 3.0), (k * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([l, a, b], dim=1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    return _xyz_to_lab(_rgb_to_xyz(rgb))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    w = mask.clamp(0.0, 1.0)
    return (x * w).sum() / (w.sum() + 1e-8)


def lab_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    pred_lab = rgb_to_lab(pred)
    target_lab = rgb_to_lab(target)
    err = (pred_lab - target_lab).abs().mean(dim=1, keepdim=True)
    return _masked_mean(err, mask)


def delta_e76_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_lab = rgb_to_lab(pred)
    target_lab = rgb_to_lab(target)
    return torch.sqrt(((pred_lab - target_lab) ** 2).sum(dim=1, keepdim=True) + 1e-12)


def delta_e76_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    return _masked_mean(delta_e76_map(pred, target), mask)


def style_distill_loss(aux: dict) -> torch.Tensor:
    """Auxiliary training loss for reference-free mode.

    Use when forward(..., ref=..., return_aux=True). Returns zero when target
    reference signals are unavailable.
    """
    if aux.get("target_style") is None or aux.get("target_ref_feat") is None:
        device = aux["pred_style"].device
        dtype = aux["pred_style"].dtype
        return torch.zeros((), device=device, dtype=dtype)

    l_style = F.l1_loss(aux["pred_style"], aux["target_style"])
    l_feat = F.l1_loss(aux["proxy_ref_feat"], aux["target_ref_feat"].detach())
    return l_style + 0.2 * l_feat
