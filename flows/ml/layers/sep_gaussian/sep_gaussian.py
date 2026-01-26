from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from .gaussian import Gaussian


class SepGaussian(ABC, torch.nn.Module):

    def __init__(
        self,
        x_channels: int,
        y_channels: int,
    ):
        super(SepGaussian, self).__init__()

        self.x_channels = x_channels
        self.y_channels = y_channels

        self.encoder = self.create_encoder(x_channels, 2 * y_channels)
        self.gaussian = Gaussian()

    @abstractmethod
    def create_encoder(self, in_channels: int, out_channels: int, **kwargs):
        return NotImplemented

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        w = self.encoder(y)
        m = w[:, :self.x_channels]
        s = F.sigmoid(w[:, self.x_channels:])
        y = self.gaussian(x, m, s)
        return y
