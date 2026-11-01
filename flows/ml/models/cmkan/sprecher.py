from typing import Optional, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from flows.ml.layers.encoders.cm_encoder import FFN
from flows.ml.layers.kan.sprecher import SprecherLayer
from flows.ml.layers.mst import MSTParameterGenerator
from .shared_cmkan import SharedCmKAN


class SprecherNet(nn.Module):
    """Conditioned Sprecher layers followed by the repository cmKAN FFN."""

    def __init__(
        self, input_dim: int, output_dim: int, layer_widths: Sequence[int] = (8, 8),
        grid_size: int = 7, spline_order: int = 3,
        psi_range: Sequence[float] = (-1, 2), xi_range: Sequence[float] = (-2, 2),
        max_shift: float = 0.5, ffn_hidden: int = 32,
        mst_features: int = 32, mst_stages: int = 3, mst_head_channels: int = 64,
        pixel_chunk_size: int = 64, checkpoint_splines: bool = True,
        block_chunk_size: Optional[int] = None,
    ):
        super().__init__()
        if not layer_widths or min(input_dim, output_dim, ffn_hidden, pixel_chunk_size) < 1:
            raise ValueError("Dimensions/chunk size must be positive and layer_widths nonempty.")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.pixel_chunk_size = pixel_chunk_size
        self.checkpoint_splines = checkpoint_splines
        dimensions = [input_dim, *layer_widths]
        self.sprecher_layers = nn.ModuleList([
            SprecherLayer(i, o, grid_size=grid_size, spline_order=spline_order,
                          psi_range=psi_range, xi_range=xi_range, max_shift=max_shift,
                          block_chunk_size=block_chunk_size)
            for i, o in zip(dimensions, dimensions[1:])
        ])
        self.parameter_sizes = tuple(layer.size for layer in self.sprecher_layers)
        self.size = sum(self.parameter_sizes)
        self.encoder = MSTParameterGenerator(
            input_dim, [block.size for layer in self.sprecher_layers for block in layer.blocks],
            mst_features, mst_stages, mst_head_channels,
        )
        self.output_ffn = FFN(layer_widths[-1], ffn_hidden, output_dim)

    def generate_parameters(self, context):
        SharedCmKAN._validate_signal(context, self.input_dim)
        return self.encoder(context)

    def apply_parameters(self, x, parameters):
        SharedCmKAN._validate_signal(x, self.input_dim)
        b, _, h, w = x.shape
        if parameters.shape != (b, self.size, h, w):
            raise ValueError("Generated curves must match the signal batch and spatial grid.")
        signals = x.permute(0, 2, 3, 1).reshape(-1, self.input_dim)
        offsets = parameters.permute(0, 2, 3, 1).reshape(-1, self.size)
        outputs = []
        for start in range(0, signals.shape[0], self.pixel_chunk_size):
            signal = signals[start:start + self.pixel_chunk_size]
            parts = offsets[start:start + self.pixel_chunk_size].split(self.parameter_sizes, -1)
            for layer, part in zip(self.sprecher_layers, parts):
                if self.checkpoint_splines and self.training and torch.is_grad_enabled():
                    signal = checkpoint(layer, signal, part, use_reentrant=False)
                else:
                    signal = layer(signal, part)
            outputs.append(signal)
        features = torch.cat(outputs).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return self.output_ffn(features)

    def forward(self, x):
        return self.apply_parameters(x, self.generate_parameters(x))

    def predictor(self, context):
        parameters = self.generate_parameters(context)
        return lambda signal: self.apply_parameters(signal, parameters)
