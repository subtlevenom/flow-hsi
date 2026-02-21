import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from flows.ml.models.cmgpd import CmGPD, LightCmGPD


class GPEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_layers:int = 3
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.x_layers = nn.ModuleList([
            LightCMEncoder(in_channels=in_channels, out_channels=in_channels)
            for _ in range(n_layers)
        ])
        self.y_layers = nn.ModuleList([
            LightCMEncoder(in_channels=in_channels, out_channels=out_channels)
            for _ in range(n_layers)
        ])
        self.gpd = LightCmGPD(
            in_channels=in_channels,
            out_channels=in_channels + out_channels,
            n_layers=n_layers,
        )

    def forward(self, src: torch.Tensor):

        z_ = []
        p_ = []
        g_ = []

        g_ = self.gpd(src)

        for x_layer, y_layer, g in zip(self.x_layers, self.y_layers, g_):
            _x = x_layer(src)
            _y = y_layer(src)

            _z = torch.cat([_x,_y], dim=1)

            p = g.log_prob(_z)

            z_.append(_z)
            p_.append(p)

        return z_, p_, g_
