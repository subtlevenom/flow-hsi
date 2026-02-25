import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.mst import MSAB


class GGPIRMSABDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 31,
    ):
        super(GGPIRMSABDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # MST++ SAB

        num_blocks=list(range(in_channels,1,-1))
        dim_head = out_channels
        dim_stage = out_channels

        self.decoder_layers = nn.Sequential()
        for i in num_blocks:
            dim_in = dim_stage * i
            dim_out = dim_stage * (i - 1)
            self.decoder_layers.append(
                MSAB(
                    dim=dim_in,
                    num_blocks=i+1,
                    dim_head=dim_head,
                    heads=i,
                ))
            self.decoder_layers.append(
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    kernel_size=1,
                    bias=False,
                ))

    def forward(self, x: torch.Tensor):
        x = rearrange(x,
                      '(b n) c h w -> b (n c) h w',
                      n=self.out_channels,
                      c=self.in_channels)
        x = self.decoder_layers(x)
        return x
