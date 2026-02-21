from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSAB
from flows.ml.layers.sep_gpd import MultivariateNormal
from .gp_encoder import GPEncoder
from .gp_corrector import GPCorrector


class GGPD(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_layers: int = 3,
    ):
        super(GGPD, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.encoder = GPEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )

        self.corrector = GPCorrector()

    def forward(self, x: torch.Tensor):
        x, p, g = self.encoder(x)
        y = self.corrector(x,p,g)
        return y[:,self.in_channels:]
