from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sep_kan_layer import SepKANLayer


class SepKAN(ABC, torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_size: int = 5,
        spline_order: int = 3,
        residual_std: float = 0.1,
        grid_range: List[float] = [0, 1],
        **kwargs,
    ):
        super(SepKAN, self).__init__()

        self.sep_kan_layer = SepKANLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )

        self.encoder = self.create_encoder(in_channels, self.sep_kan_layer.size, **kwargs)
    
    @abstractmethod
    def create_encoder(self, in_channels:int, out_channels:int, **kwargs):
        return NotImplemented

    def forward(self, x):
        w = self.encoder(x)
        x = self.sep_kan_layer(x,w)
        return x
