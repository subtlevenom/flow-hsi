from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from flows.ml.layers.sep_gpd import SepGPD, SepGPDLayer


class GPSplit(nn.Module):

    def __init__(
        self,
        split: int = 0,
        **kwargs,
    ):
        super(GPSplit, self).__init__()
        self.split = split

    def forward(
        self,
        x: torch.Tensor,
    ):
        """ 
        x: B(NC)HW -> NBCHW
        """

        B, C, H, W = x.shape
        y = rearrange(
            x,
            'b (n c) h w -> n b c h w',
            n=self.split,
            c=C // self.split,
        )

        return y
