from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
                 kernel_sizes: List[int],
                 activation: int = nn.Sigmoid):
        super().__init__()

        activation = activation or nn.Sequential
        channels = [in_channels] + out_channels

        modules = []
        for i in range(len(out_channels)):
            modules.append(
                nn.Conv2d(in_channels=channels[i],
                          out_channels=channels[i + 1],
                          kernel_size=kernel_sizes[i]))
            modules.append(activation())

        self.feed = nn.Sequential(modules[:-1])

    def forward(self, x):
        x = self.feed(x)
        return x
