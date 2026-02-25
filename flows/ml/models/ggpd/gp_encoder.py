import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from flows.ml.models.cmgpd import CmGPD, LightCmGPD, LightCmGPDLayer


class GPEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_points:int = 3
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_points = n_points

        self.x_layers = nn.ModuleList([
            LightCMEncoder(
                in_channels=in_channels,
                out_channels=in_channels+out_channels,
            ) for _ in range(n_points)
        ])

        self.gpd = LightCmGPDLayer(
            in_channels=in_channels,
            out_channels=in_channels + out_channels,
        )

        self.ffn = FFN(
            in_channels=in_channels+out_channels,
            out_channels=out_channels,
        )

    def forward(self, src: torch.Tensor):

        gy = self.gpd(src)

        x_ = []
        p_ = []
        for n in range(self.n_points):
            x = self.x_layers[n](src)
            p = gy.log_prob(x)
            x_.append(x)
            p_.append(p)

        x = torch.stack(x_,dim=1)
        p = torch.stack(p_,dim=1)
        p = torch.softmax(p, dim=1)
        # p = torch.exp(p)

        y = torch.sum(x*p, dim=1)
        y = self.ffn(y)

        return y, gy
