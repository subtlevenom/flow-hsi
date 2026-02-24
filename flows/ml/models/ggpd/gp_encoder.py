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
        n_layers:int = 3
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n_layers

        self.x_layer = CMEncoder(
            in_channels=in_channels,
            out_channels=in_channels + out_channels,
        )

        self.gpd_x = LightCmGPDLayer(
            in_channels=in_channels,
            out_channels=in_channels + out_channels,
        )

        self.gpd_y = LightCmGPDLayer(
            in_channels=in_channels,
            out_channels=in_channels + out_channels,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):

        gx = self.gpd_x(src)
        gy = self.gpd_y(src)

        x = self.x_layer(src)
        y = torch.cat([src, tgt], dim=1)

        return x, y, gx, gy 
