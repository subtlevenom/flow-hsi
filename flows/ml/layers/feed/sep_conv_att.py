from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConvAtt(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 activation: int = nn.Sigmoid):
        super(SepConvAtt, self).__init__()

        activation = activation or nn.Sequential

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=1,
                      dilation=1,
                      groups=in_channels),
            activation(inplace=True),
        )

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        w = self.attention(x)
        x = self.pointwise(x * w)
        return x
