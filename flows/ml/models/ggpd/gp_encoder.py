from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from .gpd import GPD, GPDLayer


class GPEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        **kwargs,
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.g_layer = GPDLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )

    def forward(
        self,
        src: torch.Tensor,
        x: List[torch.Tensor],
        y: List[torch.Tensor],
    ):
        g = self.g_layer(src)
        m = g.mean
        p = [g.log_prob(m+_x) for _x in x]

        p = torch.stack(p, dim=1)
        p = torch.softmax(p, dim=1)
        #p = torch.exp(p)
        y = torch.stack(y, dim=1)

        y = torch.sum(y * p, dim=1)

        return y
