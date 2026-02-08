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
    ):
        super(GPCorrector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = LightCMEncoder(
            in_channels=2*(in_channels+out_channels),
            out_channels=1,
        )
        """
        stage_ratio = 2
        dim = 3 * in_channels
        dim_stage = dim
        num_blocks = [2, 4]

        self.layers = nn.Sequential(
            FFN(
                in_channels=in_channels,
                out_channels=dim,
            ),
        )
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
            kernel_size=1,
            bias=False,
        )
        """

    def forward(self, x: torch.Tensor, y_: torch.Tensor, m: torch.Tensor):
        x = torch.cat([x,y_,m],dim=1)
        y = self.layer(x)
        return y
