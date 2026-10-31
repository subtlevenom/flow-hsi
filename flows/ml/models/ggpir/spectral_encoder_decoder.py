from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flows.ml.layers.mst import MSAB


def _positive_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class LocalBlock(nn.Module):
    """Full-resolution gated spatial refinement."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = ChannelLayerNorm(channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.spatial = nn.Conv2d(channels * 2, channels * 2, 3, padding=1,
                                 groups=channels * 2)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        signal, gate = self.spatial(self.expand(self.norm(x))).chunk(2, dim=1)
        return x + self.project(F.gelu(signal) * gate)


class SpectralEncoder(nn.Module):
    """Shared MSAB analysis/synthesis backbone, including fine feature skips."""

    def __init__(
        self, spectral_channels: int = 31, channels: int = 48,
        stage_blocks: Sequence[int] = (2, 2, 4), detail_depth: int = 2,
        checkpoint_blocks: bool = True,
    ):
        super().__init__()
        if (not _positive_integer(spectral_channels) or spectral_channels < 2
                or not _positive_integer(channels)
                or len(stage_blocks) != 3
                or not all(_positive_integer(v) for v in stage_blocks)
                or not _positive_integer(detail_depth)
                or not isinstance(checkpoint_blocks, bool)):
            raise ValueError("Require C >= 2, positive channels/detail depth, three positive "
                             "stage block counts, and boolean checkpoint_blocks.")
        self.spectral_channels = spectral_channels
        self.channels = channels
        self.stage_blocks = tuple(stage_blocks)
        self.detail_depth = detail_depth
        self.checkpoint_blocks = checkpoint_blocks
        widths = (channels, channels * 2, channels * 4)
        self.stem = nn.Conv2d(spectral_channels, channels, 3, padding=1)
        self.mask_embedding = nn.Conv2d(spectral_channels, channels, 1, bias=False)
        self.analysis = nn.ModuleList([
            self._make_stage(width, channels, 2 ** i, count)
            for i, (width, count) in enumerate(zip(widths, stage_blocks))
        ])
        self.downsample = nn.ModuleList([
            nn.Conv2d(widths[i], widths[i + 1], 4, stride=2, padding=1)
            for i in range(2)
        ])
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
            for i in (1, 0)
        ])
        self.skip_fusion = nn.ModuleList([
            nn.Conv2d(widths[i] * 2, widths[i], 1) for i in (1, 0)
        ])
        self.synthesis = nn.ModuleList([
            self._make_stage(widths[i], channels, 2 ** i, stage_blocks[i])
            for i in (1, 0)
        ])
        self.detail = nn.Sequential(*[LocalBlock(channels) for _ in range(detail_depth)])
        self.detail_fusion = nn.Conv2d(channels * 2, channels, 3, padding=1)

    @staticmethod
    def _make_stage(width, dim_head, heads, count):
        return MSAB(dim=width, dim_head=dim_head, heads=heads, num_blocks=count)

    def architecture(self):
        return {
            "version": 7, "type": "msab_unet_detail",
            "spectral_channels": self.spectral_channels, "channels": self.channels,
            "stage_blocks": list(self.stage_blocks), "widths": [
                self.channels, self.channels * 2, self.channels * 4],
            "detail_depth": self.detail_depth, "skips": "concatenation",
            "attention": "image_conditioned_spectral", "padding": "replicate_to_4",
        }

    def _run(self, module, x):
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x: torch.Tensor, band_mask: Optional[torch.Tensor] = None):
        if (x.ndim != 4 or x.shape[1] != self.spectral_channels
                or min(x.shape[0], *x.shape[-2:]) < 1):
            raise ValueError("Expected a nonempty [B, spectral_channels, H, W] cube.")
        if band_mask is not None:
            if band_mask.dtype != torch.bool or band_mask.shape != (*x.shape[:2], 1, 1):
                raise ValueError("band_mask must be boolean [B, spectral_channels, 1, 1].")
            x = x.masked_fill(band_mask, 0)
            missing = band_mask.to(x.dtype)
        else:
            missing = x.new_zeros((*x.shape[:2], 1, 1))
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % 4, 0, (-height) % 4), mode="replicate")
        stem = self.stem(x) + self.mask_embedding(missing)
        detail = self._run(self.detail, stem)
        fine = self._run(self.analysis[0], stem)
        medium = self._run(self.analysis[1], self.downsample[0](fine))
        features = self._run(self.analysis[2], self.downsample[1](medium))
        for upsample, fuse, refine, skip in zip(
                self.upsample, self.skip_fusion, self.synthesis, (medium, fine)):
            features = fuse(torch.cat((upsample(features), skip), dim=1))
            features = self._run(refine, features)
        features = stem + self.detail_fusion(torch.cat((features, detail), dim=1))
        return features[..., :height, :width]


class SpectralHead(nn.Module):
    def __init__(self, channels, spectral_channels, depth, residual):
        super().__init__()
        self.blocks = nn.Sequential(*[LocalBlock(channels) for _ in range(depth)])
        self.output = nn.Conv2d(channels, spectral_channels, 3, padding=1)
        if residual:
            nn.init.normal_(self.output.weight, std=1e-3)
            nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return self.output(self.blocks(features))


class SpectralEncoderDecoder(nn.Module):
    encoder_class = SpectralEncoder

    def __init__(
        self, head: str = "finetune", spectral_channels: int = 31, channels: int = 48,
        stage_blocks: Sequence[int] = (2, 2, 4), detail_depth: int = 2,
        head_depth: int = 2, checkpoint_blocks: bool = True,
    ):
        super().__init__()
        if head not in ("pretrain", "finetune") or not _positive_integer(head_depth):
            raise ValueError("Require head pretrain/finetune and positive integer head_depth.")
        self.head_type = head
        self.head_depth = head_depth
        self.encoder = self.encoder_class(spectral_channels, channels, stage_blocks,
                                         detail_depth, checkpoint_blocks)
        self.decoder = SpectralHead(channels, spectral_channels, head_depth, head == "finetune")

    def architecture(self):
        return {
            "encoder": self.encoder.architecture(), "head": self.head_type,
            "decoder": {"type": "spatial_residual_head", "head_depth": self.head_depth},
        }

    def forward(self, x, band_mask=None):
        output = self.decoder(self.encoder(x, band_mask))
        if self.head_type == "pretrain":
            return output
        baseline = x.masked_fill(band_mask, 0) if band_mask is not None else x
        return baseline + output
