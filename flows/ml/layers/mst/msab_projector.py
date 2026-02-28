from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum
from flows.ml.layers.mst import MSAB, MST
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm


class MSABProjector(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_blocks: List[int] = [2, 2],
    ):
        super(MSABProjector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # MST++ SAB

        self.encoder = nn.Sequential()

        dim_stage = in_channels
        dim_head = in_channels
        dim_in = in_channels

        for i, n in enumerate(num_blocks):
            dim_out = dim_stage * (i + 1)
            self.encoder.append(
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ))
            self.encoder.append(
                MSAB(
                    dim=dim_out,
                    num_blocks=n,
                    dim_head=dim_head,
                    heads=i + 1,
                ))
            dim_in = dim_out

        self.proj_out = nn.Conv2d(
            in_channels=in_channels + dim_out,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        y = self.encoder(x)
        y = self.proj_out(torch.cat([x, y], dim=1))
        return y
