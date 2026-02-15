import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSAB


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

    def forward(self, y: torch.Tensor, m: torch.Tensor, p: torch.Tensor):
        m = sum([_y*_p for _y,_p in zip(y,p)])
        p = torch.cat(p,dim=1) #.repeat_interleave(len(p))
        p = torch.sum(p,dim=1,keepdim=True)        
        m = m / p
        return m

        m = [_m[:,3:] for _m in m]
        m = torch.cat(m,dim=1)
        p = torch.cat(p,dim=1) #.repeat_interleave(len(p))
        y = torch.cat([x,m,p], dim=1)
        y = self.layers(y)
        y = self.mapping(y)
        return y

