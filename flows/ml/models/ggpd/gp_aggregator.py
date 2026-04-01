from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from flows.ml.layers.sep_kan import SepKANLayer
from .gp_projector import GPProjector


class GPAggregator(nn.Module):

    def __init__(
        self,
        grid_size: int = 7,
        spline_order: int = 5,
        residual_std: float = 0.1,
        grid_range: List[float] = [0, 1],
        **kwargs,
    ):
        super(GPAggregator, self).__init__()

        self.kan_layer = SepKANLayer(
            in_channels=1,
            out_channels=1,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ):
        """ 
        x: NBCHW
        """
        N,B,C,H,W = x.shape

        x = rearrange(x, 'n b c h w -> (n b) c h w')
        w = rearrange(w, 'n b c h w -> (n b) c h w')

        y = []
        for c in range(x.shape[1]):
            _x = x[:,c:c+1]
            _y = self.kan_layer(_x, w)
            y.append(_y)
        y = sum(y).squeeze(1)
        y = rearrange(y, '(n b) h w -> b n h w', n=N, b=B)

        return y
