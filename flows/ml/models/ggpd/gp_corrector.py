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

    def forward(
        self,
        src: torch.Tensor,
        y: torch.Tensor,
        gy: MultivariateNormal,
    ):
        z = gy.conditional_mean(y)
        z = torch.cat([y,z], dim=1)
        return z
