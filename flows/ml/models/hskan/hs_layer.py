from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from ..hsgaussian import HSGaussianLayer


class HSLayer(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_layers: int = 5,
    ):
        super(HSLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = HSGaussianLayer(in_channels, in_channels, n_layers)
        self.cube = nn.Parameter(torch.rand(1, 9*in_channels, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y0 = torch.cat([y]*9, dim=1) + self.cube
        yc = rearrange(y0, 'b (n c) h w -> (b n) c h w', n=9, c=3)
        x = torch.repeat_interleave(x, 9, dim=0)
        p = self.layer(x,yc)
        p = rearrange(p, '(b n) c h w -> b (n c) h w', n=9)
        s = torch.sum(p, dim=1, keepdim=True)
        p = p / s
        p = torch.repeat_interleave(p, self.in_channels, dim=1)
        y = y0 * p
        y = torch.split(y, 3, dim=1)
        y = sum(y)
        return y
