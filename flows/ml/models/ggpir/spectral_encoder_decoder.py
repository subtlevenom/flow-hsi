from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from flows.ml.layers.mst.MST_Plus_Plus import MS_MSA


class ChannelLayerNorm(nn.Module):
    """Normalize channels at each pixel, without image-wide spatial statistics."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class LocalBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.body = nn.Sequential(
            ChannelLayerNorm(channels),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SpectralAttentionBlock(nn.Module):
    """MST spectral attention with pre-norm on both residual paths and FFN x2."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        if heads < 1 or channels % heads:
            raise ValueError("Attention channels must be divisible by positive heads.")
        self.norm = nn.LayerNorm(channels)
        self.attention = MS_MSA(dim=channels, dim_head=channels // heads, heads=heads)
        self.feed_forward = LocalBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x.movedim(1, -1)
        features = features + self.attention(self.norm(features))
        return self.feed_forward(features.movedim(-1, 1))


class SpectralEncoder(nn.Module):
    def __init__(
        self, spectral_channels: int = 31,
        channels: Sequence[int] = (32, 64, 128, 192),
        depths: Sequence[int] = (1, 2, 2),
        heads: Sequence[int] = (2, 4, 6),
    ):
        super().__init__()
        if (spectral_channels < 2 or len(channels) != 4 or len(depths) != 3
                or len(heads) != 3 or min(channels) < 1 or min(depths) < 1):
            raise ValueError("Require C >= 2, four positive widths and three positive depths/heads.")
        self.spectral_channels = spectral_channels
        self.channels = tuple(channels)
        self.depths = tuple(depths)
        self.heads = tuple(heads)
        self.stem = nn.Conv2d(spectral_channels, channels[0], 3, padding=1)
        self.mask_embedding = nn.Conv2d(spectral_channels, channels[0], 1, bias=False)
        self.local = LocalBlock(channels[0])
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1),
                *[SpectralAttentionBlock(channels[i + 1], heads[i])
                  for _ in range(depths[i])],
            )
            for i in range(3)
        ])

    def architecture(self) -> dict:
        return {
            "version": 1, "spectral_channels": self.spectral_channels,
            "channels": list(self.channels), "depths": list(self.depths),
            "heads": list(self.heads), "ffn_expansion": 2,
            "normalization": "per_pixel_channel_layernorm",
        }

    def forward(
        self, x: torch.Tensor, band_mask: Optional[torch.Tensor] = None,
    ):
        if x.ndim != 4 or x.shape[1] != self.spectral_channels or min(x.shape[-2:]) < 1:
            raise ValueError("Expected a nonempty [B, spectral_channels, H, W] cube.")
        features = self.stem(x)
        if band_mask is not None:
            if band_mask.dtype != torch.bool or band_mask.shape != (*x.shape[:2], 1, 1):
                raise ValueError("band_mask must be boolean [B, spectral_channels, 1, 1].")
            mask_features = band_mask.to(x.dtype)
        else:
            mask_features = x.new_zeros((*x.shape[:2], 1, 1))
        features = features + self.mask_embedding(mask_features)
        features = self.local(features)
        pyramid = [features]
        for stage in self.stages:
            features = stage(features)
            pyramid.append(features)
        return pyramid


class ReconstructionHead(nn.Module):
    """Bottleneck-only convolutional decoder: no shallow skips or source bypass."""

    def __init__(
        self, in_channels: int, spectral_channels: int,
        channels: Sequence[int] = (96, 48, 32),
    ):
        super().__init__()
        if len(channels) != 3 or min(channels) < 1:
            raise ValueError("Reconstruction head requires three positive widths.")
        widths = (in_channels, *channels)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(widths[i], widths[i + 1], 1), LocalBlock(widths[i + 1]))
            for i in range(3)
        ])
        self.output = nn.Conv2d(channels[-1], spectral_channels, 3, padding=1)

    def forward(self, pyramid):
        x = pyramid[-1]
        for stage, reference in zip(self.stages, reversed(pyramid[:-1])):
            x = stage(F.interpolate(x, size=reference.shape[-2:],
                                    mode="bilinear", align_corners=False))
        return self.output(x)


class RestorationStage(nn.Module):
    def __init__(self, in_channels: int, channels: int, depth: int, heads: int):
        super().__init__()
        self.project = nn.Conv2d(in_channels, channels, 1)
        self.skip = nn.Conv2d(channels, channels, 1)
        self.blocks = nn.Sequential(
            *[SpectralAttentionBlock(channels, heads) for _ in range(depth)]
        ) if depth else LocalBlock(channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.blocks(self.project(x) + self.skip(skip))


class RestorationHead(nn.Module):
    def __init__(
        self, channels: Sequence[int], heads: Sequence[int],
        spectral_channels: int,
    ):
        super().__init__()
        self.decoder2 = RestorationStage(channels[3], channels[2], 2, heads[1])
        self.decoder1 = RestorationStage(channels[2], channels[1], 1, heads[0])
        self.decoder0 = RestorationStage(channels[1], channels[0], 0, 1)
        self.output = nn.Conv2d(channels[0], spectral_channels, 3, padding=1)
        # Near identity, but decoder gradients are nonzero from the first update.
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, pyramid):
        e0, e1, e2, e3 = pyramid
        x = self.decoder2(e3, e2)
        x = self.decoder1(x, e1)
        return self.output(self.decoder0(x, e0))


class SpectralEncoderDecoder(nn.Module):
    def __init__(
        self, head: str = "restoration", spectral_channels: int = 31,
        channels: Sequence[int] = (32, 64, 128, 192),
        depths: Sequence[int] = (1, 2, 2),
        heads: Sequence[int] = (2, 4, 6),
        reconstruction_channels: Sequence[int] = (96, 48, 32),
    ):
        super().__init__()
        if head not in ("pretrain", "restoration"):
            raise ValueError("head must be 'pretrain' or 'restoration'.")
        self.head_type = head
        self.encoder = SpectralEncoder(spectral_channels, channels, depths, heads)
        self.decoder = (
            ReconstructionHead(channels[-1], spectral_channels, reconstruction_channels)
            if head == "pretrain"
            else RestorationHead(channels, heads, spectral_channels)
        )

    def forward(self, x: torch.Tensor, band_mask: Optional[torch.Tensor] = None):
        result = self.decoder(self.encoder(x, band_mask))
        return result if self.head_type == "pretrain" else x + result
