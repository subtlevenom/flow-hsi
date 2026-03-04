from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from .gpd import GPD, GPDLayer


class GPCorrector(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super(GPCorrector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj = FFN(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        *v: List[torch.Tensor],
    ):
        y = v[:2]
        p = v[2:]

        p = torch.stack(p, dim=1)
        p = torch.softmax(p, dim=1)

        y = torch.stack(y, dim=1)
        y = torch.sum(y * p, dim=1)

        y = self.proj(y)

        return y
