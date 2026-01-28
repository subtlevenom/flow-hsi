from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .hs_gaussian import HSGaussian
from flows.ml.layers.encoders import SGEncoder, CMEncoder, LightCMEncoder
from .hs_gaussian_layer import HSGaussianLayer

class HSGaussianMixture(nn.Module):

    def __init__(
        self,
        x_channels: int,
        y_channels: int,
        g_channels: int,
        out_channels: int,
    ):
        super(HSGaussianMixture, self).__init__()

        self.x_channels = x_channels
        self.y_channels = y_channels
        self.g_channels = g_channels
        self.out_channels = out_channels

        self.layer = HSGaussianLayer(x_channels, y_channels, g_channels)
        self.mixture = nn.Conv2d(
            in_channels=g_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        x = self.layer(x,y)
        x = self.mixture(x)
        return x
