from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.sep_gpd import MultivariateNormal
from flows.ml.layers.sep_gpd import SepGPD, SepGPDLayer


class GPReplicate(nn.Module):

    def __init__(
        self,
        n: int = 0,
        dim: int=0,
        **kwargs,
    ):
        super(GPReplicate, self).__init__()
        self.n = n
        self.dim=dim

    def forward(
        self,
        x: torch.Tensor,
    ):
        """ 
        x: BCHW -> NBCHW
        """

        x = torch.stack([x]*self.n, dim=self.dim)
        return x
