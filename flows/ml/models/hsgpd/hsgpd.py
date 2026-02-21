from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSAB
from flows.ml.layers.sep_gpd import MultivariateNormal
from flows.ml.models.cmkan import CmKAN
from flows.ml.models.ggpd import GGPD


class HSGPD(nn.Module):

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
        super(HSGPD, self).__init__()

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

        self.n = 9
        self.distorter = LightCMEncoder(in_channels, self.n * out_channels)
        self.corrector = LightCMEncoder(self.n * in_channels, out_channels)


    def forward(self, x: torch.Tensor):
        
        dx = self.distorter(x)

        x0 = x.repeat(self.n, dim=1)
        xd = x0 + dx

        xd = rearrange(xd, 'b (n c) h w -> (b n) c h w', n = self.n)
        xd = self.cmkan(xd)
        xd = rearrange(xd, '(b n) c h w -> b (n c) h w', n = self.n)

        y = self.corrector(xd)

        return y

