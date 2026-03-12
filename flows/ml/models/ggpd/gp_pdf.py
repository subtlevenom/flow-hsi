from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from flows.ml.layers.sep_gpd import SepGPD, SepGPDLayer


class GPPdf(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_layers: int = 1,
        **kwargs,
    ):
        super(GPPdf, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers

        self.gpd = SepGPD(
            dim=in_channels,
            num_layers=num_layers,
            **kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ):
        """ 
        x: NBCHW
        w: NBCHW
        """
        g = self.gpd(w)
        p = [_g.log_prob(_x) for _x,_g in zip(x,g)]

        p = torch.stack(p, dim=0)
        p = torch.softmax(p, dim=0)

        return p
