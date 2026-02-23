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

        self.dx_layer = LightCMEncoder(
            self.in_channels,
            self.n * (self.in_channels + self.out_channels),
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

        y0 = torch.cat([src, tgt], dim=1)
        x0 = self.x_layer(src)

        dx = self.dx_layer(src)
        dx = rearrange(dx, 'b (n c) h w -> b n c h w', n = self.n)

        px_ = []
        py_ = []
        x_ = []
        y_ = []
        
        for i in range(self.n):
            yd = y0 + dx[:,i]
            xd = x0 + dx[:, i]

            py = gy.log_prob(yd)
            px = gx.log_prob(xd)

            py_.append(py)
            px_.append(px)
            y_.append(yd)
            x_.append(xd)

        py = torch.softmax(torch.cat(py_, dim=1), dim=1)
        px = torch.softmax(torch.cat(px_, dim=1), dim=1)

        y = torch.stack(y_, dim=1)
        y = torch.sum(y * py.unsqueeze(2),dim=1, keepdim=False)

        x = torch.stack(x_, dim=1)
        x = torch.sum(x * px.unsqueeze(2),dim=1, keepdim=False)

        return x, y, px, py, gx, gy 
