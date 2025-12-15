import torch
import torch.nn as nn
import torch.nn.functional as F
from flows.ml.layers.blocks import SConv2d
from einops import rearrange


class HSConv(nn.Module):

    def __init__(
        self,
        in_channels: int = 31,
        out_channels: int = 31,
    ):
        super(HSConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        mid_channels = [in_channels, in_channels, in_channels]

        self.l1 = SConv2d(in_channels=in_channels, out_channels=mid_channels[0])
        self.l2 = SConv2d(in_channels=mid_channels[0], out_channels=mid_channels[1])
        self.l3 = SConv2d(in_channels=mid_channels[1], out_channels=mid_channels[2])
        self.l4 = SConv2d(in_channels=mid_channels[2], out_channels=out_channels)

    def forward(self, x: torch.Tensor):
        x1 = self.l1(x)
        x2 = self.l2(x + x1)
        x3 = self.l3(x1 + x2)
        x4 = self.l4(x2 + x3)
        return x4
