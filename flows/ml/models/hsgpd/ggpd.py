from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSAB
from flows.ml.layers.sep_gpd import MultivariateNormal


class GGPD(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_layers: int = 3,
    ):
        super(GGPD, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        stage_ratio = 2
        dim = in_channels
        dim_stage = dim

        """
        self.layers = nn.Sequential()
        for stage_num_blocks in num_blocks:
            self.layers.append(
                MSAB(
                    dim=dim_stage,
                    num_blocks=stage_num_blocks,
                    dim_head=dim,
                    heads=dim_stage // dim,
                ),
            )
            self.layers.append(
                nn.Conv2d(
                    in_channels=dim_stage,
                    out_channels=stage_ratio * dim_stage,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
            )
            dim_stage *= stage_ratio

        self.mapping = nn.Conv2d(
            in_channels=(stage_ratio ** len(num_blocks)) * dim,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        """

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

