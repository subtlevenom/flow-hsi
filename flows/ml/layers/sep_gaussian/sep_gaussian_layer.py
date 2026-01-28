from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .sep_gaussian import SepGaussian


class SepGaussianLayer(ABC, nn.Module):

    def __init__(
        self,
        x_channels: int,
        y_channels: int,
        g_channels: int,
    ):
        super(SepGaussianLayer, self).__init__()

        self.x_channels = x_channels
        self.y_channels = y_channels
        self.g_channels = g_channels

        self.layers = nn.ModuleList([
            self.create_gaussian(
                x_channels,
                y_channels,
            ) for _ in range(g_channels)
        ])

    @abstractmethod
    def create_gaussian(
        self,
        x_channels: int,
        y_channels: int,
        **kwargs,
    ) -> SepGaussian:
        return NotImplemented

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y = torch.concat([g(x, y) for g in self.layers], dim=1)
        return y
