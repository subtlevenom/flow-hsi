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
        self.g_layers = nn.ModuleList([
            GPGaussian(x_channels=in_channels, y_channels=out_channels)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, tgt: torch.Tensor):

        x_ = []
        y_ = []
        m_ = []
        p_ = []

        for x_layer, g_layer in zip(self.x_layers, self.g_layers):
            _x = x_layer(x)
            g = g_layer(_x)
            # x,y
            m = g.mean
            # y|x
            gy_x = g.conditional_distribution(_x)
            y_x = gy_x.mean
            z = torch.cat([_x,y_x], dim=1)
            p = g.prob(z)

            x_.append(_x)
            y_.append(y_x)
            m_.append(m)
            p_.append(p.unsqueeze(1))

        return x_, y_, m_, p_
