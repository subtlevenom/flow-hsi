from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.kan.shared_spline import SharedSplineLayer
from flows.ml.layers.mst import MSTParameterGenerator


class SharedCmKAN(nn.Module):
    """Spatially conditioned composition of layer-shared spline neurons."""

    def __init__(
        self, in_channels: int, out_channels: int,
        hidden_channels: Sequence[int] = (), grid_size: int = 7,
        spline_order: int = 3, grid_range: Sequence[float] = (0, 1),
        conditioning_channels: int = 8, encoder_type: str = "light",
        pixel_chunk_size: int = 1024, checkpoint_splines: bool = True,
        mst_features: int = 32, mst_stages: int = 3, mst_head_channels: int = 64,
    ):
        super().__init__()
        if conditioning_channels < 1 or pixel_chunk_size < 1:
            raise ValueError("conditioning_channels and pixel_chunk_size must be positive.")
        if encoder_type not in ("light", "full", "mst"):
            raise ValueError("encoder_type must be 'light', 'full' or 'mst'.")
        dimensions = [in_channels, *hidden_channels, out_channels]
        self.in_channels, self.out_channels = in_channels, out_channels
        self.pixel_chunk_size = pixel_chunk_size
        self.checkpoint_splines = checkpoint_splines
        self.spline_layers = nn.ModuleList([
            SharedSplineLayer(i, o, grid_size, spline_order, grid_range)
            for i, o in zip(dimensions, dimensions[1:])
        ])
        self.parameter_sizes = tuple(layer.size for layer in self.spline_layers)
        self.size = sum(self.parameter_sizes)
        self.encoder_type = encoder_type
        if encoder_type == "mst":
            # MST++ consumes the raw spectrum; its heads emit every layer's vector.
            self.encoder = MSTParameterGenerator(
                in_channels, self.parameter_sizes, mst_features, mst_stages, mst_head_channels,
            )
            return
        # This projection is only for context generation; neurons consume x directly.
        self.conditioning_projection = nn.Conv2d(in_channels, conditioning_channels, 1)
        encoder = LightCMEncoder if encoder_type == "light" else CMEncoder
        self.encoder = encoder(conditioning_channels, self.size)
        nn.init.zeros_(self.encoder.conv_reproj.pointwise2.weight)
        nn.init.zeros_(self.encoder.conv_reproj.pointwise2.bias)

    def generate_parameters(self, context: torch.Tensor):
        self._validate_signal(context, self.in_channels)
        if self.encoder_type == "mst":
            return self.encoder(context)
        h, w = context.shape[-2:]
        # The original wavelet generator requires multiples of four, while
        # the pointwise mapping itself supports arbitrary positive image sizes.
        padded = F.pad(context, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
        if min(padded.shape[-2:]) < 8:
            padded = F.pad(padded, (0, max(0, 8 - padded.shape[-1]),
                                   0, max(0, 8 - padded.shape[-2])), mode="replicate")
        return self.encoder(self.conditioning_projection(padded))[..., :h, :w]

    @staticmethod
    def _validate_signal(x, channels):
        if x.ndim != 4 or x.shape[1] != channels or min(x.shape[-2:]) < 1:
            raise ValueError(f"Expected [B, {channels}, H, W] with positive spatial dimensions.")

    def apply_parameters(self, x: torch.Tensor, parameters: torch.Tensor):
        self._validate_signal(x, self.in_channels)
        if parameters.shape != (x.shape[0], self.size, *x.shape[-2:]):
            raise ValueError("Generated parameters must match the signal batch and spatial grid.")
        b, _, h, w = x.shape
        signals = x.permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        offsets = parameters.permute(0, 2, 3, 1).reshape(-1, self.size)
        outputs = []
        for start in range(0, signals.shape[0], self.pixel_chunk_size):
            signal = signals[start:start + self.pixel_chunk_size]
            chunks = offsets[start:start + self.pixel_chunk_size].split(self.parameter_sizes, dim=-1)
            for layer, generated in zip(self.spline_layers, chunks):
                if self.checkpoint_splines and self.training and torch.is_grad_enabled():
                    signal = checkpoint(layer, signal, generated, use_reentrant=False)
                else:
                    signal = layer(signal, generated)
            outputs.append(signal)
        return torch.cat(outputs).reshape(b, h, w, self.out_channels).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor):
        return self.apply_parameters(x, self.generate_parameters(x))

    def predictor(self, context: torch.Tensor):
        parameters = self.generate_parameters(context)
        return lambda signal: self.apply_parameters(signal, parameters)
