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
        x_channels: int = 3,
        y_channels: int = 3,
        out_channels: int = 3,
        n_layers: int = 3,
    ):
        super(HSLayer, self).__init__()

        self.x_channels = x_channels
        self.y_channels = y_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        z_channels = x_channels + y_channels

        self.layer = HSGaussianLayer(y_channels, x_channels + y_channels, n_layers)
        self.proj = FFN(in_channels=z_channels + n_layers, out_channels=out_channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = torch.cat([x,y], dim=1)
        y = self.layer(y,z)
        y = torch.concat([z, y], dim=1)
        y = self.proj(y)
        return y
