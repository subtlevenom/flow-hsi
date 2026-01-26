from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from ..hsgaussian import HSGaussianMixture


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
        self.w = 3 * 3

        self.layer = HSGaussianMixture(in_channels, in_channels, n_layers)
        self.offset = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.w * in_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        o = self.offset(x)
        y0 = torch.cat([y]*self.w, dim=1) + o
        x0 = torch.cat([x]*self.w, dim=1)
        yc = rearrange(y0, 'b (n c) h w -> (b n) c h w', n=self.w, c=3)
        xc = rearrange(x0, 'b (n c) h w -> (b n) c h w', n=self.w, c=3)
        p = self.layer(xc,yc)
        p = rearrange(p, '(b n) c h w -> b (n c) h w', n=9)
        s = torch.sum(p, dim=1, keepdim=True)
        p = p / s
        p = torch.repeat_interleave(p, self.in_channels, dim=1)
        y = y0 * p
        y = torch.split(y, 3, dim=1)
        y = sum(y)
        return y
