import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Blend explicit pixel-wise color transform and nonlinear residual mapping."""

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
        return torch.sigmoid(out)

    def forward(self, feat: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        explicit = self._pixel_color_transform(feat, src)
        spline = torch.sigmoid(self.spline_head(feat))
        g = self.gate(feat)
        return g * spline + (1.0 - g) * explicit


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

        self.ref_enc = RefStyleEncoder(
            in_channels=3,
            feat_channels=channels,
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

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = HybridColorHead(channels)

    def forward(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.dim() == 3:
            src = src.unsqueeze(0)
        if ref.dim() == 3:
            ref = ref.unsqueeze(0)

        illum_feat, illum_map = self.illum(src)
        feat = self.src_body(src) + self.src_shortcut(src)
        feat = self.fusion(torch.cat([feat, illum_feat], dim=1))
        style = self.ref_enc(ref)

        state = None
        for layer in self.osc_layers:
            feat, state = layer(feat, style, illum_feat * illum_map, state=state)

        return self.head(feat, src)
