import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .MST_Plus_Plus import MSAB


class HSEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 31,
        out_channels: int = 3,
    ):
        super(HSEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # MST++ SAB

        num_blocks=list(range(2,out_channels+1))
        dim_head = in_channels
        dim_stage = in_channels

        self.encoder_layers = nn.Sequential()
        for i in num_blocks:
            dim_in = dim_stage * (i - 1)
            dim_out = dim_stage * i
            self.encoder_layers.append(
                MSAB(
                    dim=dim_in,
                    num_blocks=i,
                    dim_head=dim_head,
                    heads=i - 1,
                ))
            self.encoder_layers.append(
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    kernel_size=1,
                    bias=False,
                ))

    def forward(self, x: torch.Tensor):
        x = self.encoder_layers(x)
        x = rearrange(x,
                      'b (n c) h w -> (b n) c h w',
                      n=self.in_channels,
                      c=self.out_channels)
        return x
