from typing import Sequence

import torch
from torch import nn

from flows.ml.layers.mst.MST_Plus_Plus import FeedForward
from .spectral_encoder_decoder import (
    ChannelLayerNorm, SpectralEncoder, SpectralEncoderDecoder, _positive_integer,
)


class ConvolutionalRestorationBlock(nn.Module):
    """Local mixer with approximately the same parameter budget as one MSAB block."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        )
        self.norm = ChannelLayerNorm(channels)
        self.ffn = FeedForward(channels)

    def forward(self, x):
        x = x + self.mix(x) + self.spatial(x)
        return x + self.ffn(self.norm(x).movedim(1, -1)).movedim(-1, 1)


class ConvolutionalSpectralEncoder(SpectralEncoder):
    @staticmethod
    def _make_stage(width, dim_head, heads, count):
        return nn.Sequential(*[ConvolutionalRestorationBlock(width) for _ in range(count)])

    def architecture(self):
        return {**super().architecture(), "type": "convolutional_unet_detail",
                "attention": "none", "mixer_expansion": 2}


class ConvolutionalSpectralEncoderDecoder(SpectralEncoderDecoder):
    """Replace MSAB only; preserve pyramid, skips, detail, and task heads."""

    encoder_class = ConvolutionalSpectralEncoder


class GlobalCalibratedSpectralEncoder(SpectralEncoder):
    def __init__(self, spectral_channels=31, channels=48, stage_blocks=(2, 2, 4),
                 detail_depth=2, checkpoint_blocks=True):
        super().__init__(spectral_channels, channels, stage_blocks,
                         detail_depth, checkpoint_blocks)
        self.calibration = nn.Sequential(
            nn.Linear(spectral_channels * 2, channels), nn.GELU(),
            nn.Linear(channels, channels * 2),
        )
        nn.init.normal_(self.calibration[-1].weight, std=1e-3)
        nn.init.zeros_(self.calibration[-1].bias)

    def architecture(self):
        return {**super().architecture(), "type": "global_calibrated_msab_unet_detail",
                "calibration": "source_band_mean_std_feature_film"}

    def forward(self, x, band_mask=None):
        features = super().forward(x, band_mask)
        visible = x.masked_fill(band_mask, 0) if band_mask is not None else x
        # Float32 statistics avoid unstable variance estimates under mixed precision.
        statistics = visible.float().flatten(2)
        mean = statistics.mean(-1)
        std = statistics.var(-1, unbiased=False).add(1e-8).sqrt()
        gain, bias = self.calibration(torch.cat((mean, std), dim=1)).chunk(2, dim=1)
        return features * (1 + gain.tanh()[..., None, None]) + bias[..., None, None]


class GlobalCalibratedSpectralEncoderDecoder(SpectralEncoderDecoder):
    """Add source-only image-level calibration to the otherwise unchanged MSAB U-Net."""

    encoder_class = GlobalCalibratedSpectralEncoder


class IterativeSpectralEncoder(nn.Module):
    def __init__(self, spectral_channels=31, channels=48, stage_blocks=(2, 2, 4),
                 detail_depth=2, checkpoint_blocks=True, iterations=3):
        super().__init__()
        if not _positive_integer(iterations) or iterations < 2:
            raise ValueError("iterations must be an integer >= 2.")
        self.channels = channels
        self.iterations = iterations
        self.stages = nn.ModuleList([
            SpectralEncoder(spectral_channels, channels, stage_blocks,
                            detail_depth, checkpoint_blocks) for _ in range(iterations)
        ])
        self.refinement_heads = nn.ModuleList([
            nn.Conv2d(channels, spectral_channels, 3, padding=1)
            for _ in range(iterations - 1)
        ])
        for head in self.refinement_heads:
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    def architecture(self):
        return {"version": 7, "type": "iterative_msab_unet_detail",
                "iterations": self.iterations, "weight_sharing": False,
                "stage": self.stages[0].architecture(),
                "refinement": "intermediate_spectral_residual"}

    def forward(self, x, band_mask=None):
        features = self.stages[0](x, band_mask)
        current = x.masked_fill(band_mask, 0) if band_mask is not None else x
        for head, stage in zip(self.refinement_heads, self.stages[1:]):
            current = current + head(features)
            # Later stages see model estimates, not the hidden input or target bands.
            features = stage(current)
        return features, current


class IterativeSpectralEncoderDecoder(SpectralEncoderDecoder):
    """Independent residual restoration stages; only the final task head is phase-specific."""

    def __init__(self, head="finetune", spectral_channels=31, channels=48,
                 stage_blocks: Sequence[int] = (2, 2, 4), detail_depth=2,
                 head_depth=2, checkpoint_blocks=True, iterations=3):
        # Supply the normal factory a stage constructor without registering duplicate modules.
        self.iterations = iterations
        super().__init__(head, spectral_channels, channels, stage_blocks,
                         detail_depth, head_depth, checkpoint_blocks)

    def encoder_class(self, *args):
        return IterativeSpectralEncoder(*args, iterations=self.iterations)

    def forward(self, x, band_mask=None):
        features, current = self.encoder(x, band_mask)
        output = self.decoder(features)
        return output if self.head_type == "pretrain" else current + output
