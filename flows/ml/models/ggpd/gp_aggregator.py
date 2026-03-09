from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from flows.ml.layers.sep_gpd import SepGPD, SepGPDLayer


class GPAggregator(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_layers: int = 1,
        **kwargs,
    ):
        super(GPAggregator, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers

        self.gpd = SepGPD(
            dim=in_channels,
            num_layers=num_layers,
            **kwargs
        )

    def forward(
        self,
        x: Union[List[torch.Tensor], torch.Tensor],
        w: List[torch.Tensor],
        y: List[torch.Tensor],
    ):
        """ 
        x: NBCHW or 1BCHW
        y,w: NBCHW
        """
        g = self.gpd(w)
        
        if len(x.shape) == len(y.shape):
            p = [_g.log_prob(_x) for _x,_g in zip(x,g)]
        else:
            p = [_g.log_prob(x) for _g in g]

        p = torch.stack(p, dim=0)
        p = torch.softmax(p, dim=0)

        y = torch.sum(y * p, dim=0)

        return y
