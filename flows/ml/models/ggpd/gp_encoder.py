from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum
from .gpd import create_encoder


class GPEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: List[int] = [3, 3],
        alg: str = 'mix',
        num_blocks: List[int] = [2, 2],
        num_points: int = 7,
        **kwargs,
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_points = num_points

        self.projectors = nn.ModuleList([
            create_encoder(
                in_channels=in_channels,
                out_channels=sum(out_channels),
                alg=alg,
                num_blocks=num_blocks,
                **kwargs,
            ) for _ in range(num_points)
        ])

    def forward(self, x: torch.Tensor):
        y_ = []
        for projector in self.projectors:
            y = projector(x)
            y_.append(y)

        y_ = torch.stack(y_,dim=0) # nbchw
        if len(self.out_channels) > 1:
            y_ = torch.split(y_, list(self.out_channels), dim=2)

        return y_
