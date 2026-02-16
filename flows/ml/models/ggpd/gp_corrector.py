from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSAB
from flows.ml.layers.gpd_gaussian import MultivariateNormal


class GPCorrector(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_blocks = [2, 4]
    ):
        super(GPCorrector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        stage_ratio = 2
        dim = in_channels
        dim_stage = dim

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

    def forward(self, y: List[torch.Tensor], log_p: List[torch.Tensor], g: List[MultivariateNormal] ):
        C = y[0].shape[1]

        log_q = [_g.log_prob(_g.mean).unsqueeze(1) for _g in g]
        p = torch.cat(log_p, dim=1)
        q = torch.cat(log_q, dim=1)
        w = torch.stack([p, q], dim=0)
        w = torch.softmax(w,dim=0)
        p0 = w[0]
        q0 = w[1]

        p = torch.repeat_interleave(p0, C, dim=1)
        q = torch.repeat_interleave(q0, C, dim=1)

        y = torch.cat(y,dim=1)
        m = torch.cat([_g.mean[:,3:] for _g in g], dim=1)
        y = y * p + m * q

        p = torch.softmax(2 * q0 / (p0 + q0) - 1,dim=1)
        p = torch.repeat_interleave(p, C, dim=1)
        y = y * p

        y = rearrange(y, 'b (n c) w h -> b n c w h', n = len(log_p))
        y = torch.sum(y, dim=1, keepdim=False)
        return y

