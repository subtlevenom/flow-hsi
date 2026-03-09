from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import einops
from .multivariate_normal import MultivariateNormal
from .sep_gpd_layer import SepGPDLayer


class SepGPD(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        num_layers: int = 3,
        **kwargs,
    ):
        super(SepGPD, self).__init__()

        self.dim = dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [SepGPDLayer(
                dim=dim,
                **kwargs,
            ) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        g_ = [_g(_x) for _g, _x in zip(self.layers, x)]
        return g_

    def cond_distrib_predictor(self, x: torch.Tensor):
        g_ = self.forward(x)
        return lambda y, dims: [
            g.conditional_distribution(y, dims) for g in g_
        ]

    def log_prob_predictor(self, x: torch.Tensor):
        g_ = self.forward(x)
        return lambda y: torch.cat([g.log_prob(y) for g in g_], dim=1)
