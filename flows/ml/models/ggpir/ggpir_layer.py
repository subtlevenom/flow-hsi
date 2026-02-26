from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from .hs_gaussian_layer import HSGaussianLayer


class GGPIRLayer(nn.Module):

    def __init__(
        self,
        x_channels: int = 3,
        y_channels: int = 3,
        g_channels: int = 3,
        out_channels: int = 3,
    ):
        super(GGPIRLayer, self).__init__()

        self.x_channels = x_channels
        self.y_channels = y_channels
        self.g_channels = g_channels
        self.out_channels = out_channels

        self.layer = HSGaussianLayer(x_channels, y_channels, g_channels)
        self.proj = nn.Conv2d(
            in_channels=x_channels + g_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor=None):
        y = x if y is None else y
        y = self.layer(x,y)
        y = torch.concat([x, y], dim=1)
        y = self.proj(y)
        return y
