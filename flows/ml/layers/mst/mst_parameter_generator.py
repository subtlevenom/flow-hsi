from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .MST_Plus_Plus import MST


class MSTParameterGenerator(nn.Module):
    """MST++ feature backbone with per-layer parameter heads.

    Adapted from MST++ (Cai et al., CVPRW 2022,
    https://github.com/caiyuanhao1998/MST-plus-plus, MIT License) via this
    repository's ``MST_Plus_Plus.py``: a stem plus ``stages`` cascaded
    spectral-wise-attention U-shaped MST blocks with a global residual. Instead
    of the RGB->31-band mapping, the backbone keeps ``features`` channels and
    1x1 heads emit each layer's flattened generated parameter vector.
    """

    def __init__(
        self, in_channels: int, parameter_sizes: Sequence[int],
        features: int = 32, stages: int = 3, head_channels: int = 64,
    ):
        super().__init__()
        if min(in_channels, features, stages, head_channels) < 1 or not parameter_sizes:
            raise ValueError("MST generator dimensions must be positive.")
        self.stem = nn.Conv2d(in_channels, features, 3, padding=1, bias=False)
        self.body = nn.Sequential(*[
            MST(in_dim=features, out_dim=features, dim=features,
                stage=2, num_blocks=[1, 1, 1])
            for _ in range(stages)
        ])
        self.shared_head = nn.Sequential(nn.Conv2d(features, head_channels, 1), nn.GELU())
        self.heads = nn.ModuleList([nn.Conv2d(head_channels, size, 1)
                                    for size in parameter_sizes])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # Two MST down/up-sampling levels need multiples of four; replicate
        # padding (unlike reflect) also supports 1x1 inputs.
        x = F.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
        x = self.stem(x)
        return (self.body(x) + x)[..., :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.shared_head(self.features(x))
        return torch.cat([head(hidden) for head in self.heads], dim=1)
