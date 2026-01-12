import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.models.cmkan import LightCmKAN
from ..hsgaussian import HSGaussianLayer


class HSEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 31,
        out_channels: int = 3,
    ):
        super(HSEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        all_channels = in_channels * out_channels

        self.weights = nn.Sequential(
            nn.Conv2d(
                in_channels=all_channels,
                out_channels=all_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=all_channels,
                out_channels=all_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
            ),
        )

        self.layer = HSGaussianLayer(in_channels=1, out_channels=1)

    def forward(self, x: torch.Tensor):
        x = torch.cat([x] * self.out_channels, dim=1)
        w = self.weights(x)
        w = rearrange(w, 'b c h w -> (b c) 1 h w')
        w = self.layer(w)
        w = rearrange(w,
                      '(b c) 1 h w -> b c h w',
                      c=self.in_channels * self.out_channels)

        x = x * (1. + w)
        x = rearrange(x,
                      'b (n c) h w -> (b n) c h w',
                      n=self.in_channels,
                      c=self.out_channels)
        return x
