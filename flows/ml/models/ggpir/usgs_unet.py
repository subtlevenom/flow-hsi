from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    # Keep at least two channels per group.
    groups = min(8, max(1, channels // 2))
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3,
                padding=1, bias=False
            ),
            group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, 3,
                padding=1, bias=False
            ),
            group_norm(out_channels),
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.project = nn.Conv2d(in_channels, out_channels, 1)
        self.fuse = ResidualBlock(
            out_channels + skip_channels, out_channels
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.project(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """
    Three spatial downsampling levels.

    Returns a full-resolution tensor with out_channels channels.
    Explicit skip sizes support odd image dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        zero_output: bool = False,
    ):
        super().__init__()

        b = base_channels

        self.encoder0 = ResidualBlock(in_channels, b)
        self.encoder1 = DownBlock(b, 2 * b)
        self.encoder2 = DownBlock(2 * b, 4 * b)
        self.bottleneck = DownBlock(4 * b, 8 * b)

        self.decoder2 = UpBlock(8 * b, 4 * b, 4 * b)
        self.decoder1 = UpBlock(4 * b, 2 * b, 2 * b)
        self.decoder0 = UpBlock(2 * b, b, b)

        self.output = nn.Conv2d(b, out_channels, 1)

        if zero_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        b = self.bottleneck(e2)

        d2 = self.decoder2(b, e2)
        d1 = self.decoder1(d2, e1)
        d0 = self.decoder0(d1, e0)

        return self.output(d0)


class GaussianInjection(nn.Module):
    """
    Per-pixel Gaussian spectral modulation.

    Inputs:
        spectrum: [B, C, H, W]
        features: [B, F, H, W]

    Predicted parameter groups:
        center, width, gain amplitude, offset amplitude

    Each parameter group has shape [B, K, H, W].
    """

    def __init__(
        self,
        feature_channels: int,
        spectral_channels: int = 31,
        num_gaussians: int = 6,
        wavelengths: Optional[Sequence[float]] = None,
        gain_scale: float = 0.1,
        offset_scale: float = 0.05,
        min_width: float = 0.03,
        max_width: float = 0.50,
    ):
        super().__init__()

        if spectral_channels < 2:
            raise ValueError("At least two spectral channels are required.")
        if num_gaussians < 1:
            raise ValueError("num_gaussians must be positive.")
        if not 0.0 < min_width < max_width:
            raise ValueError("Require 0 < min_width < max_width.")
        if gain_scale < 0.0 or offset_scale < 0.0:
            raise ValueError("Modulation scales must be nonnegative.")

        self.num_gaussians = num_gaussians
        self.gain_scale = gain_scale
        self.offset_scale = offset_scale
        self.min_width = min_width
        self.max_width = max_width

        if wavelengths is None:
            # Assumes uniformly spaced wavelength samples.
            wavelength_grid = torch.linspace(
                0.0, 1.0, spectral_channels
            )
        else:
            wavelength_grid = torch.as_tensor(
                wavelengths, dtype=torch.float32
            )

            if wavelength_grid.shape != (spectral_channels,):
                raise ValueError(
                    "wavelengths must contain one value per channel."
                )
            if not bool(torch.isfinite(wavelength_grid).all()):
                raise ValueError("Wavelengths must be finite.")
            if not bool(
                (wavelength_grid[1:] > wavelength_grid[:-1]).all()
            ):
                raise ValueError(
                    "Wavelengths must be strictly increasing."
                )

            wavelength_grid = (
                wavelength_grid - wavelength_grid[0]
            ) / (
                wavelength_grid[-1] - wavelength_grid[0]
            )

        self.register_buffer(
            "wavelength_grid",
            wavelength_grid.reshape(1, 1, spectral_channels, 1, 1),
        )

        hidden1 = max(8, feature_channels // 2)
        hidden2 = max(4, feature_channels // 4)

        self.reduce = nn.Sequential(
            nn.Conv2d(feature_channels, hidden1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden1, hidden2, 1),
            nn.SiLU(inplace=True),
        )

        self.parameters_out = nn.Conv2d(
            hidden2, 4 * num_gaussians, 1
        )

        # Give Gaussian centers distinct initial positions.
        # Zero amplitudes make the injection initially an identity.
        with torch.no_grad():
            k = num_gaussians

            self.parameters_out.weight[:2 * k].normal_(
                mean=0.0, std=0.01
            )
            self.parameters_out.bias[:2 * k].zero_()

            initial_centers = torch.linspace(0.1, 0.9, k)
            self.parameters_out.bias[:k].copy_(
                torch.logit(initial_centers)
            )

            self.parameters_out.weight[2 * k:].zero_()
            self.parameters_out.bias[2 * k:].zero_()

    def forward(self, spectrum, features):
        parameters = self.parameters_out(self.reduce(features))

        center_raw, width_raw, gain_amp, offset_amp = (
            parameters.chunk(4, dim=1)
        )

        centers = torch.sigmoid(center_raw)
        widths = self.min_width + (
            self.max_width - self.min_width
        ) * torch.sigmoid(width_raw)

        # Gaussian evaluation in float32 also supports mixed precision.
        wavelengths = self.wavelength_grid.float()
        centers = centers.float().unsqueeze(2)
        widths = widths.float().unsqueeze(2)

        # [B, K, C, H, W]
        basis = torch.exp(
            -0.5 * ((wavelengths - centers) / widths).square()
        )

        # Sum over Gaussian components, preserving spectral channels.
        gain_field = (
            gain_amp.float().unsqueeze(2) * basis
        ).sum(dim=1)

        offset_field = (
            offset_amp.float().unsqueeze(2) * basis
        ).sum(dim=1)

        gamma = self.gain_scale * torch.tanh(gain_field)
        beta = self.offset_scale * torch.tanh(offset_field)

        gamma = gamma.to(dtype=spectrum.dtype)
        beta = beta.to(dtype=spectrum.dtype)

        return spectrum * (1.0 + gamma) + beta


class ParallelHyperspectralUNets(nn.Module):
    """
    Cascaded parallel Unet-CM and Unet-H.

    Input:  [B, 31, H, W]
    Output: [B, 31, H, W]

    Unet-H evolves its own feature state independently of the CM state.
    The branches interact through the Gaussian injection heads.
    """

    def __init__(
        self,
        spectral_channels: int = 31,
        num_cascades: int = 3,
        cm_base_channels: int = 32,
        hyper_channels: int = 24,
        hyper_base_channels: int = 24,
        num_gaussians: int = 6,
        wavelengths: Optional[Sequence[float]] = None,
        gain_scale: float = 0.1,
        offset_scale: float = 0.05,
    ):
        super().__init__()

        if num_cascades < 1:
            raise ValueError("num_cascades must be positive.")

        self.spectral_channels = spectral_channels

        self.hyper_stem = ResidualBlock(
            spectral_channels, hyper_channels
        )

        self.cm_stages = nn.ModuleList([
            UNet(
                in_channels=spectral_channels,
                out_channels=spectral_channels,
                base_channels=cm_base_channels,
                zero_output=True,
            )
            for _ in range(num_cascades)
        ])

        self.hyper_stages = nn.ModuleList([
            UNet(
                in_channels=hyper_channels,
                out_channels=hyper_channels,
                base_channels=hyper_base_channels,
                zero_output=False,
            )
            for _ in range(num_cascades)
        ])

        self.injections = nn.ModuleList([
            GaussianInjection(
                feature_channels=hyper_channels,
                spectral_channels=spectral_channels,
                num_gaussians=num_gaussians,
                wavelengths=wavelengths,
                gain_scale=gain_scale,
                offset_scale=offset_scale,
            )
            for _ in range(num_cascades)
        ])

    def forward(self, x, return_intermediate: bool = False):
        if x.ndim != 4 or x.shape[1] != self.spectral_channels:
            raise ValueError(
                "Expected [B, spectral_channels, H, W]."
            )

        y = x
        h = self.hyper_stem(x)
        intermediate = []

        for cm_stage, hyper_stage, injection in zip(
            self.cm_stages,
            self.hyper_stages,
            self.injections,
        ):
            # Conceptually parallel branch updates.
            provisional = y + cm_stage(y)
            h = h + hyper_stage(h)

            # Inject before the next CM cascade.
            y = injection(provisional, h)

            if return_intermediate:
                intermediate.append(y)

        if return_intermediate:
            return y, intermediate

        return y


if __name__ == "__main__":
    model = ParallelHyperspectralUNets(
        spectral_channels=31,
        num_cascades=3,
        cm_base_channels=32,
        hyper_channels=24,
        hyper_base_channels=24,
        num_gaussians=6,
    )

    # Odd dimensions are supported.
    x = torch.rand(1, 31, 65, 67)

    model.eval()
    with torch.no_grad():
        y, stages = model(x, return_intermediate=True)

    assert y.shape == x.shape
    assert len(stages) == 3

    # Identity initialization, up to floating-point precision.
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))