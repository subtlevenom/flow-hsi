import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.gpd_gaussian import MultivariateNormal
from .gp_gaussian import GPGaussian


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
            LightCMEncoder(in_channels=in_channels, out_channels=in_channels)
            for _ in range(n_layers)
        ])
        self.g_layers = nn.ModuleList([
            GPGaussian(x_channels=in_channels, y_channels=out_channels)
            for _ in range(n_layers)
        ])

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):

        x_ = []
        p_ = []
        g_ = []

        for x_layer, y_layer, g_layer in zip(self.x_layers, self.y_layers, self.g_layers):
            _x = x_layer(src)
            _y = y_layer(src)
            # x,y
            gxy:MultivariateNormal = g_layer(src)

            _z = torch.cat([_x,_y], dim=1)
            p = gxy.log_prob(_z)

            x_.append(_z)
            p_.append(p)
            g_.append(gxy)

        return x_, p_, g_
