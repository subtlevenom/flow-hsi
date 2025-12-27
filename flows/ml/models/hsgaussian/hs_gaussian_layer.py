from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from torch import nn
from .hs_gaussian import HSGaussian


class HSGaussianLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(HSGaussianLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.ModuleList(
            [HSGaussian(in_channels) for _ in range(out_channels)])

    def forward(self, x:torch.Tensor):
        y = torch.concat([g(x) for g in self.layers], dim=1)
        return y
