import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from flows.ml.layers.kan import SepKANLayer2D


class HSKANHead(nn.Module):

    def __init__(self,
                 in_channels: int = 31,
                 out_channels: int = 50,):
        super(HSKANHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.unshuffle = nn.PixelUnshuffle(4)
        self.proj = nn.Conv2d(
                in_channels=4*4*3,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, s: torch.Tensor, x: torch.Tensor):
        x = self.unshuffle(x)
        x = self.proj(x)
        x = s + F.tanh(x)
        return torch.clamp(x, min=0.0, max=1.0)
