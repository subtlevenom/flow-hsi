from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSAB
from flows.ml.layers.sep_gpd import MultivariateNormal
from .gp_aggregator import GPEncoder
from .gp_x_projector import GPProjector


class GGPD(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_layers: int = 3,
        num_blocks: List[int] = [2, 2],
        num_points: int = 7,
        alg: str = 'mix',
    ):
        super(GGPD, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.projector = GPProjector(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_points=num_points,
        )

        self.encoder = GPEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            alg=alg,
        )

    def forward(self, x: torch.Tensor):
        _x, _y = self.projector(x)
        y = self.encoder(x, _x, _y)
        return y
