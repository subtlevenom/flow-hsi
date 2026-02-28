from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum
from flows.ml.layers.mst import MSAB, MST
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from .msab_projector import MSABProjector
from flows.ml.layers.encoders import LightCMEncoder


class MSABCMProjector(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_blocks: List[int] = [2, 2],
    ):
        super(MSABCMProjector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # MST++ SAB

        self.msab = MSABProjector(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
        )
        self.cm = LightCMEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.mix = nn.Conv2d(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        x1 = self.msab(x)
        x2 = self.cm(x)
        y = self.mix(torch.cat([x1, x2], dim=1))
        return y
