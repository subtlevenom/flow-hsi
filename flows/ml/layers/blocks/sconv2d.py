from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: List,
        kernel_size:int=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.basis = nn.Parameter(torch.rand(self.out_channels, self.out_channels))
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x):
        c = self.conv(x)
        c = F.softmax(c,dim=1)
        x = torch.einsum('bchw,cs->bshw', c, self.basis)
        return x
