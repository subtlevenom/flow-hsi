from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import einops
from .multivariate_normal import MultivariateNormal
from .sep_gpd_layer import SepGPDLayer


class SepGPD(ABC, torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        **kwargs,
    ):
        super(SepGPD, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            self.create_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            ) for _ in range(num_layers)
        ])

    @abstractmethod
    def create_layer(self, in_channels: int, out_channels: int,
                     s_range: List[int], **kwargs) -> SepGPDLayer:
        return NotImplemented

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        g_ = []
        for _g, _x in zip(self.layers, x):
            g_.append(_g(_x))
        return g_

    def cond_distrib_predictor(self, x: torch.Tensor):
        g_ = self.forward(x)
        return lambda y, dims: [
            g.conditional_distribution(y, dims) for g in g_
        ]

    def log_prob_predictor(self, x: torch.Tensor):
        g_ = self.forward(x)
        return lambda y: torch.cat([g.log_prob(y) for g in g_], dim=1)
