from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum
from .gpd import create_encoder


class GPProjector(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        alg: str = 'mix',
        num_blocks: List[int] = [2, 2],
        **kwargs,
    ):
        super(GPProjector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.projector = create_encoder(
            in_channels=in_channels,
            out_channels=out_channels,
            alg=alg,
            num_blocks=num_blocks,
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        x = self.projector(x)
        return x
