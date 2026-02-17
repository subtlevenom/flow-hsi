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
        p_ = []
        g_ = []
        ss = []

        for x_layer, g_layer in zip(self.x_layers, self.g_layers):
            _x = x_layer(x)
            # x,y
            gxy,s = g_layer(x)
            # y|x
            gy_x:MultivariateNormal = gxy.conditional_distribution(_x)
            # p(y|x)
            y_x = gy_x.mean
            z = torch.cat([_x,y_x], dim=1)
            p = gxy.log_prob(z)

            x_.append(_x)
            y_.append(y_x)
            p_.append(p.unsqueeze(1))
            g_.append(gxy)
            ss.append(s[0,:,20,20])

        return x_, y_, p_, g_
