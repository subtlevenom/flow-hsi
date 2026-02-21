from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSAB
from flows.ml.layers.sep_gpd import MultivariateNormal
from ..cmkan import CmKAN


class HSKan(nn.Module):

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
        super(HSKan, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cmkan = CmKAN(
            in_channels=in_channels,
            out_channels=out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )

        self.distorter = LightCMEncoder(in_channels, out_channels)
        self.corrector = LightCMEncoder(in_channels, out_channels)


    def forward(self, x: List[torch.Tensor], p: List[torch.Tensor], g: List[MultivariateNormal] ):
        C = x[0].shape[1]

        p = torch.cat(p, dim=1)
        p = torch.softmax(p,dim=1)
        p = torch.repeat_interleave(p, C, dim=1)

        y = torch.cat(x,dim=1)
        y = y * p

        y = rearrange(y, 'b (n c) w h -> b n c w h', n = len(x))
        y = torch.sum(y, dim=1, keepdim=False)

        return y

