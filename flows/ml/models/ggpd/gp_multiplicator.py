from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from flows.ml.layers.sep_gpd import SepGPD, SepGPDLayer


class GPMultiplicator(nn.Module):

    def __init__(
        self,
        split: int = 0,
        **kwargs,
    ):
        super(GPMultiplicator, self).__init__()
        self.split = split

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
    ):
        """ 
        x: NBCHW
        w: NB1HW
        """

        y = torch.sum(x * p, dim=0)

        if self.split > 0:
            B,C,H,W = y.shape
            y = rearrange(
                y,
                'b (n s) h w -> n b s h w',
                n=self.split,
                s=C // self.split,
            )

        return y
