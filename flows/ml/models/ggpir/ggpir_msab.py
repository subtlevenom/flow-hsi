from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.models.ggpd import GGPD
from .hs_encoder import HSEncoder
from .hs_layer import HSLayer


class HSNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 31,
        out_channels: int = 3,
        n_iters: int = 0,
    ):
        super(HSNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        g_channels = 1 + out_channels + in_channels

        self.encoder = HSEncoder(
            in_channels=in_channels,
            out_channels=in_channels,
            n_iter=n_iters,
        )
        self.layer = HSLayer(
            x_channels=in_channels,
            y_channels=in_channels,
            g_channels=g_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y = self.encoder(src)
        y = self.layer(x, y)
        return y
