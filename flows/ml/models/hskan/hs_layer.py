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
        n_layers: int = 7,
    ):
        super(HSLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = HSGaussianLayer(in_channels, n_layers)
        self.norm = LayerNorm(in_channels)
        self.ffn = FFN(in_channels=n_layers + 2 * in_channels,
                       out_channels=in_channels)

    def forward(self, x: torch.Tensor):
        z = self.norm(x)
        y = self.layer(x)
        y = torch.concat([x, z, y], dim=1)
        y = self.ffn(y)
        return y
