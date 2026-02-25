import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum
from flows.ml.layers.mst import MSAB


class GPProjector(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super(GPProjector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # MST++ SAB

        dim_head = out_channels
        num_blocks = (in_channels - out_channels) // dim_head
        dim_stage = dim_head * (num_blocks + 1)

        self.layers = nn.Sequential()

        # in proj
        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=dim_stage,
                kernel_size=1,
                bias=False,
            ))

        # msab
        for i in range(1, num_blocks+1):
            dim_in = dim_stage - dim_head * (i - 1)
            dim_out = dim_stage - dim_head * i
            self.layers.append(
                MSAB(
                    dim=dim_in,
                    num_blocks=num_blocks+1 - (i-1),
                    dim_head=dim_head,
                    heads=num_blocks + 1 - (i - 1),
                ))
            self.layers.append(
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ))

        # out proj
        self.layers.append(
            nn.Conv2d(
                in_channels=dim_head,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ))

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return x
