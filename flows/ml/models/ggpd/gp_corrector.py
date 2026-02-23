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

    def __init__(self, in_channels, out_channels):
        super(GPCorrector, self).__init__()


    def forward(self, x: List[torch.Tensor], gx, gy):
        return x
