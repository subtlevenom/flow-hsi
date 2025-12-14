from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SSum2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: List,
        kernel_size:int=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_a = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.conv_x = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.conv_x = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x):
        a = self.conv(x)
        a = rearrange(a,
                      'b (s c) h w -> b s c h w',
                      s=self.out_channels,
                      c=self.in_channels)
        x = torch.einsum('bschw,bchw->bshw', a, x) + b
        return x
