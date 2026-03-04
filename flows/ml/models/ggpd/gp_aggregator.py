from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from .gpd import GPD, GPDLayer


class GPAggregator(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_layers: int = 1,
        **kwargs,
    ):
        super(GPAggregator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.gpd = GPD(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            alg = None,
            **kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        w: List[torch.Tensor],
        y: List[torch.Tensor],
    ):
        g = self.gpd(w)
        p = [_g.log_prob(x) for _g in g]

        p = torch.stack(p, dim=1)
        p = torch.softmax(p, dim=1)

        y = torch.stack(y, dim=1)
        y = torch.sum(y * p, dim=1)

        return y
