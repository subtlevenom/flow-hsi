from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .hs_gaussian import HSGaussian
from flows.ml.layers.encoders import SGEncoder, CMEncoder, LightCMEncoder

class HSGaussianMixture(nn.Module):

    def __init__(
        self,
        x_channels: int,
        y_channels: int,
        n_layers: int,
    ):
        super(HSGaussianMixture, self).__init__()

        self.x_channels = x_channels
        self.y_channels = y_channels
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [HSGaussian(x_channels,y_channels) for _ in range(n_layers)])
        self.mixture = LightCMEncoder(x_channels, n_layers)

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        z = torch.concat([x,y],dim=1)
        y = torch.concat([g(y,z) for g in self.layers], dim=1)
        a = self.mixture(x)
        y = torch.sum(a * y, dim=1, keepdim=True)
        return y
