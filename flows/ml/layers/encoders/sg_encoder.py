import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.mw_isp import DWTForward, RCAGroup, DWTInverse, seq


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FFN(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """

    def __init__(self, in_channels, hidden_channels=None, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.pointwise1 = nn.Conv2d(in_channels,
                                    hidden_channels,
                                    kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_channels,
                                   hidden_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   dilation=1,
                                   groups=hidden_channels)
        self.pointwise2 = nn.Conv2d(hidden_channels,
                                    out_channels,
                                    kernel_size=1)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x

# SG Encoder

class SGEncoder(torch.nn.Module):
    """ Input features BxCxHxW """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SGEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            LayerNorm(out_channels),
            FFN(out_channels, out_channels=out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x
