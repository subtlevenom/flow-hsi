from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum
from flows.ml.layers.mst import MSAB, MST
from .msab_projector import MSABProjector


class GPProjector(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: List[int] = [3, 3],
        num_blocks: List[int] = [2, 2],
        num_points: int = 7,
    ):
        super(GPProjector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_points = num_points

        self.projectors = nn.ModuleList([
            MSABProjector(
                in_channels=in_channels,
                out_channels=sum(out_channels),
                num_blocks=num_blocks,
            ) for _ in range(num_points)
        ])

    def forward(self, x: torch.Tensor):
        x_ = []
        y_ = []
        for projector in self.projectors:
            v = projector(x)
            _x, _y = torch.split(v, list(self.out_channels), dim=1)
            x_.append(_x)
            y_.append(_y)

        return x_, y_
