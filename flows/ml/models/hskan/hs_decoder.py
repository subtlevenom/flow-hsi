import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.models.cmkan import LightCmKAN
from ..hsgaussian import HSGaussianLayer


class HSDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 31,
    ):
        super(HSDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        all_channels = in_channels * out_channels

        self.layer = HSGaussianLayer(in_channels=1, out_channels=1)

        self.proj = nn.Conv2d(
            in_channels=all_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

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
                groups=out_channels,
            ),
        )

    def forward(self, x: torch.Tensor):
        x = rearrange(x,
                      '(b n) c h w -> b (n c) h w',
                      n=self.out_channels,
                      c=self.in_channels)

        w = self.weights(x)
        w = rearrange(w, 'b c h w -> (b c) 1 h w')
        w = self.layer(w)
        w = rearrange(w,
                      '(b c) 1 h w -> b c h w',
                      c=self.in_channels * self.out_channels)
        x = x * (1. + w)
        x1 = x[:,:self.out_channels]
        x2 = x[:,self.out_channels:2 * self.out_channels]
        x3 = x[:,2 * self.out_channels:]
        x = (x1 + x2 + x3) / 3.
        return x
