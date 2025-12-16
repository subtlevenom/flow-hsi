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

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features,
                                    hidden_features,
                                    kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_features,
                                   hidden_features,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   dilation=1,
                                   groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features,
                                    out_features,
                                    kernel_size=1)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x


class HSKANEncoder(torch.nn.Module):
    """
    sepconv replace conv_out to reduce GFLOPS
    """

    def __init__(self, in_channels:int=31, out_channels:int=90):
        super().__init__()

        MID_CHANNELS = 3 * in_channels

        self.in_proj = FFN(in_features=in_channels, out_features=MID_CHANNELS)
        self.norm1 = LayerNorm(MID_CHANNELS)

        N = 128
        self.basis = nn.Parameter(torch.rand(1, MID_CHANNELS, N))

        self.q = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.k = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.v = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)

        self.norm2 = LayerNorm(MID_CHANNELS)

        self.conv_reproj = FFN(in_features=MID_CHANNELS,
                               out_features=out_channels)

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape

        x = self.in_proj(x)
        x = self.norm1(x)

        # basis coeff
        q = self.q(x)
        k = self.k(self.basis)
        v = self.v(self.basis)

        q = rearrange(q, 'b c h w -> b c (h w)')

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        a = (q.transpose(-2, -1) @ k).transpose(-2, -1)
        a = F.relu(a)

        y = v @ a

        y = rearrange(y, 'b c (h w) -> b c h w', h=H, w=W)

        # back projection
        x = self.norm2(y)
        x = self.conv_reproj(x)

        return x
