from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSAB
from flows.ml.layers.sep_gpd import MultivariateNormal


class GPCorrector(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_points: int):
        super(GPCorrector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_points = n_points

        self.dx_layer = LightCMEncoder(
            self.in_channels,
            n_points * (self.in_channels + self.out_channels),
        )

    def forward(
        self,
        src: torch.Tensor,
        x: torch.Tensor,
        gx: MultivariateNormal,
    ):

        dx = self.dx_layer(src)
        dx = rearrange(dx, 'b (n c) h w -> b n c h w', n=self.n_points)

        p_ = []
        x_ = []

        for n in range(self.n_points):
            z = x + dx[:,n]
            p = gx.log_prob(z)

            x_.append(z)
            p_.append(p)
        
        p_ = torch.stack(p_, dim=1)
        p_ = torch.softmax(p_, dim=1)

        x_ = torch.stack(x_, dim=1)

        x_ = torch.sum(x_ * p_, dim=1)

        return x_
