from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSAB
from flows.ml.layers.sep_gpd import MultivariateNormal


class GPCorrector(nn.Module):

    def __init__(self):
        super(GPCorrector, self).__init__()

    def forward(self, x: List[torch.Tensor], p: List[torch.Tensor], g: List[MultivariateNormal] ):
        C = x[0].shape[1]

        p = torch.cat(p, dim=1)
        p = torch.softmax(p,dim=1)
        p = torch.repeat_interleave(p, C, dim=1)

        y = torch.cat(x,dim=1)
        y = y * p

        y = rearrange(y, 'b (n c) w h -> b n c w h', n = len(x))
        y = torch.sum(y, dim=1, keepdim=False)

        return y

