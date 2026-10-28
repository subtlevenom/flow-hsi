from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _positive_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class ChannelLayerNorm(nn.Module):
    """Normalize channels at each pixel, without image-wide statistics."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class LocalBlock(nn.Module):
    """Local spatial mixing and content-gated cross-channel transformation."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.norm = ChannelLayerNorm(channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.spatial = nn.Conv2d(
            channels * 2, channels * 2, 3, padding=dilation,
            dilation=dilation, groups=channels * 2,
        )
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal, gate = self.spatial(self.expand(self.norm(x))).chunk(2, dim=1)
        return x + self.project(F.gelu(signal) * gate)


class PixelQueryBlock(nn.Module):
    """One pixel-conditioned query attends only to that pixel's RF tokens."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(channels)
        self.token_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key_value = nn.Linear(channels, channels * 2)
        self.project = nn.Linear(channels, channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        count, token_count, channels = tokens.shape
        head_dim = channels // self.heads
        q = self.query(self.query_norm(query)).reshape(count, self.heads, 1, head_dim)
        kv = self.key_value(self.token_norm(tokens))
        k, v = kv.reshape(count, token_count, 2, self.heads, head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        query = query + self.project(attended.reshape(count, channels))
        return query + self.ffn(self.ffn_norm(query))


class ResidualConvBlock(nn.Module):
    """ResNet-style spatial features without batch or image-wide normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + 0.1 * self.body(x)


class GatedScaleConnection(nn.Module):
    def __init__(self, source_channels: int, target_channels: int):
        super().__init__()
        self.project = nn.Conv2d(source_channels, target_channels, 1)
        self.gate = nn.Conv2d(2 * target_channels, target_channels, 1)
        nn.init.normal_(self.gate.weight, std=1e-3)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, target, source):
        source = self.project(source)
        gate = torch.sigmoid(self.gate(torch.cat((target, source), dim=1)))
        return target + gate * source


class SpectralEncoder(nn.Module):
    """Connected full/half/quarter residual streams and per-pixel token fusion."""

    def __init__(
        self, spectral_channels: int = 31, channels: int = 64,
        branch_depths: Sequence[int] = (3, 3, 4),
        branch_channels: Sequence[int] = (48, 96, 192),
        detail_depth: int = 2, fusion_depth: int = 2, heads: int = 4,
        pixel_chunk_size: int = 4096, checkpoint_blocks: bool = True,
    ):
        super().__init__()
        if (not _positive_integer(spectral_channels) or spectral_channels < 2
                or not all(_positive_integer(v) for v in (
                    channels, detail_depth, fusion_depth, heads, pixel_chunk_size))
                or channels % heads
                or len(branch_depths) != 3 or len(branch_channels) != 3
                or not all(_positive_integer(v) for v in (*branch_depths, *branch_channels))
                or not isinstance(checkpoint_blocks, bool)):
            raise ValueError(
                "Require C >= 2, three positive integer branch widths/depths, "
                "positive widths/depths/chunk size, heads dividing channels, "
                "and boolean checkpoint_blocks."
            )
        self.spectral_channels = spectral_channels
        self.channels = channels
        self.branch_depths = tuple(branch_depths)
        self.branch_channels = tuple(branch_channels)
        self.detail_depth = detail_depth
        self.fusion_depth = fusion_depth
        self.heads = heads
        self.pixel_chunk_size = pixel_chunk_size
        self.checkpoint_blocks = checkpoint_blocks
        self.stem = nn.Conv2d(spectral_channels, branch_channels[0], 3, padding=1)
        self.pixel_projection = nn.Conv2d(spectral_channels, channels, 1)
        self.mask_embedding = nn.Conv2d(spectral_channels, branch_channels[0], 1, bias=False)
        self.mask_projection = nn.Conv2d(branch_channels[0], channels, 1, bias=False)
        self.scale_inputs = nn.ModuleList([
            nn.Identity(),
            nn.Conv2d(branch_channels[0], branch_channels[1], 1),
            nn.Conv2d(branch_channels[0], branch_channels[2], 1),
        ])
        self.branches = nn.ModuleList([
            nn.Sequential(*[ResidualConvBlock(width) for _ in range(depth)])
            for depth, width in zip(branch_depths, branch_channels)
        ])
        self.fine_to_medium = GatedScaleConnection(branch_channels[0], branch_channels[1])
        self.fine_to_broad = GatedScaleConnection(branch_channels[0], branch_channels[2])
        self.medium_to_broad = GatedScaleConnection(branch_channels[1], branch_channels[2])
        self.detail = nn.Sequential(*[
            ResidualConvBlock(branch_channels[0]) for _ in range(detail_depth)
        ])
        self.token_projections = nn.ModuleList([
            nn.Conv2d(width, channels, 1) for width in (*branch_channels, branch_channels[0])
        ])
        self.branch_embedding = nn.Parameter(torch.empty(5, channels))
        nn.init.normal_(self.branch_embedding, std=0.02)
        self.fusion = nn.ModuleList([
            PixelQueryBlock(channels, heads) for _ in range(fusion_depth)
        ])
        self.output_norm = nn.LayerNorm(channels)

    def architecture(self) -> dict:
        return {
            "version": 4, "type": "connected_multiscale_query_transformer",
            "spectral_channels": self.spectral_channels, "channels": self.channels,
            "branch_depths": list(self.branch_depths),
            "branch_channels": list(self.branch_channels),
            "scales": [1, 2, 4], "connections": "gated_fine_to_medium_and_broad_medium_to_broad",
            "detail_depth": self.detail_depth, "fusion_depth": self.fusion_depth,
            "heads": self.heads, "ffn_expansion": 2,
            "normalization": "none_in_residual_encoders_per_pixel_in_fusion",
            "downsampling": "average_pool_fixed_grid", "padding": "replicate_right_bottom_to_4",
            "residual_scale": 0.1,
        }

    def _local_path(self, path, x):
        for block in path:
            x = checkpoint(block, x, use_reentrant=False) if (
                self.checkpoint_blocks and self.training and torch.is_grad_enabled()
            ) else block(x)
        return x

    def _fuse(self, tokens):
        tokens = tokens + self.branch_embedding
        query = tokens[:, 0] + tokens[:, -1]
        for block in self.fusion:
            query = block(query, tokens)
        return self.output_norm(query)

    def forward(self, x: torch.Tensor, band_mask: Optional[torch.Tensor] = None):
        if (x.ndim != 4 or x.shape[1] != self.spectral_channels
                or min(x.shape[0], *x.shape[-2:]) < 1):
            raise ValueError("Expected a nonempty [B, spectral_channels, H, W] cube.")
        if band_mask is not None:
            if band_mask.dtype != torch.bool or band_mask.shape != (*x.shape[:2], 1, 1):
                raise ValueError("band_mask must be boolean [B, spectral_channels, 1, 1].")
            # Mask here as well as in the pipeline: hidden values never reach any path.
            x = x.masked_fill(band_mask, 0)
            mask_features = band_mask.to(x.dtype)
        else:
            mask_features = x.new_zeros((*x.shape[:2], 1, 1))
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % 4, 0, (-height) % 4), mode="replicate")
        missing = self.mask_embedding(mask_features)
        stem = self.stem(x) + missing
        pixel = self.pixel_projection(x) + self.mask_projection(missing)
        fine = self._local_path(self.branches[0], stem)
        medium_input = self.scale_inputs[1](F.avg_pool2d(stem, 2))
        medium_input = self.fine_to_medium(medium_input, F.avg_pool2d(fine, 2))
        medium = self._local_path(self.branches[1], medium_input)
        broad_input = self.scale_inputs[2](F.avg_pool2d(stem, 4))
        broad_input = self.fine_to_broad(broad_input, F.avg_pool2d(fine, 4))
        broad_input = self.medium_to_broad(broad_input, F.avg_pool2d(medium, 2))
        broad = self._local_path(self.branches[2], broad_input)
        detail = self._local_path(self.detail, stem)
        projected = [project(feature) for project, feature in
                     zip(self.token_projections, (fine, medium, broad, detail))]
        features = [
            pixel, projected[0],
            F.interpolate(projected[1], scale_factor=2, mode="bilinear", align_corners=False),
            F.interpolate(projected[2], scale_factor=4, mode="bilinear", align_corners=False),
            projected[3],
        ]
        batch, channels, padded_height, padded_width = pixel.shape
        # Chunk only the independent per-pixel regression, never the convolutional context.
        flattened = [feature.movedim(1, -1).reshape(-1, channels) for feature in features]
        outputs = []
        for start in range(0, batch * padded_height * padded_width, self.pixel_chunk_size):
            tokens = torch.stack(
                [feature[start:start + self.pixel_chunk_size] for feature in flattened], dim=1,
            )
            fused = checkpoint(self._fuse, tokens, use_reentrant=False) if (
                self.checkpoint_blocks and self.training and torch.is_grad_enabled()
            ) else self._fuse(tokens)
            outputs.append(fused)
        return torch.cat(outputs).reshape(
            batch, padded_height, padded_width, channels,
        ).movedim(-1, 1)[..., :height, :width]


class SpectralHead(nn.Module):
    def __init__(self, channels: int, spectral_channels: int, depth: int, residual: bool):
        super().__init__()
        self.blocks = nn.Sequential(*[LocalBlock(channels) for _ in range(depth)])
        self.output = nn.Conv2d(channels, spectral_channels, 1)
        if residual:
            nn.init.normal_(self.output.weight, std=1e-3)
            nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return self.output(self.blocks(features))


class SpectralEncoderDecoder(nn.Module):
    """Keep the existing factory entrypoint; both task heads now run at full resolution."""

    def __init__(
        self, head: str = "finetune", spectral_channels: int = 31, channels: int = 64,
        branch_depths: Sequence[int] = (3, 3, 4),
        branch_channels: Sequence[int] = (48, 96, 192),
        detail_depth: int = 2, fusion_depth: int = 2, heads: int = 4,
        head_depth: int = 2, pixel_chunk_size: int = 4096,
        checkpoint_blocks: bool = True,
    ):
        super().__init__()
        if head not in ("pretrain", "finetune") or not _positive_integer(head_depth):
            raise ValueError("Require head pretrain/finetune and a positive integer head_depth.")
        self.head_type = head
        self.head_depth = head_depth
        self.encoder = SpectralEncoder(
            spectral_channels, channels, branch_depths, branch_channels,
            detail_depth, fusion_depth, heads, pixel_chunk_size, checkpoint_blocks,
        )
        self.decoder = SpectralHead(channels, spectral_channels, head_depth, head == "finetune")

    def architecture(self) -> dict:
        return {
            "encoder": self.encoder.architecture(), "head": self.head_type,
            "decoder": {"type": "full_resolution", "head_depth": self.head_depth},
        }

    def forward(self, x: torch.Tensor, band_mask: Optional[torch.Tensor] = None):
        features = self.encoder(x, band_mask)
        if self.head_type == "pretrain":
            return self.decoder(features)
        baseline = x.masked_fill(band_mask, 0) if band_mask is not None else x
        return baseline + self.decoder(features)
