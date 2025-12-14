import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HSEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 31,
        out_channels: int = 3,
    ):
        super(HSEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        all_channels = in_channels * out_channels

        self.weights = nn.Conv2d(
            in_channels=all_channels,
            out_channels=all_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )

    def forward(self, x: torch.Tensor):
        x = torch.cat([x] * self.out_channels, dim=1)
        x = x * (1. + self.weights(x)**2)
        x = rearrange(x,
                      'b (n c) h w -> (b n) c h w',
                      n=self.in_channels,
                      c=self.out_channels)
        return x
