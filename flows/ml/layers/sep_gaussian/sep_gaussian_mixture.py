from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .sep_gaussian_layer import SepGaussianLayer


class SepGaussianMixture(ABC, nn.Module):

    def __init__(
        self,
        x_channels: int,
        y_channels: int,
        g_channels: int,
        out_channels: int,
    ):
        super(SepGaussianMixture, self).__init__()

        self.x_channels = x_channels
        self.y_channels = y_channels
        self.g_channels = g_channels
        self.out_channels = out_channels

        self.layer = self.create_gaussian_layer(
            x_channels,
            y_channels,
            g_channels,
        )
        self.mixture = nn.Conv2d(
            in_channels=g_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    @abstractmethod
    def create_gaussian_layer(
        self,
        x_channels: int,
        y_channels: int,
        g_channels: int,
        **kwargs,
    ) -> SepGaussianLayer:
        return NotImplemented

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.layer(x, y)
        x = self.mixture(x)
        return x
