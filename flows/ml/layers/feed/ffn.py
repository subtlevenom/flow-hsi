from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """

    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or in_channels
        self.pointwise1 = nn.Conv2d(in_channels,
                                    mid_channels,
                                    kernel_size=1)
        self.depthwise = nn.Conv2d(mid_channels,
                                   mid_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   dilation=1,
                                   groups=mid_channels)
        self.pointwise2 = nn.Conv2d(mid_channels,
                                    out_channels,
                                    kernel_size=1)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x
