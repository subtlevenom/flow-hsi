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

        self.encoder = self.create_encoder(y_channels, 2 * x_channels + 1)
        self.gaussian = Gaussian()

    @abstractmethod
    def create_encoder(self, in_channels: int, out_channels: int, **kwargs):
        return NotImplemented

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        w = self.encoder(y)
        a = w[:, :1]
        m = w[:, 1:1+self.x_channels]
        s = w[:, 1+self.x_channels:]
        y = self.gaussian(x, a, m, s)
        return y

    def predictor(self, x):
        w = self.encoder(x)
        a = w[:, :1]
        m = w[:, 1:1+self.x_channels]
        s = w[:, 1+self.x_channels:]
        return lambda y: self.gaussian(y, a, m, s)
