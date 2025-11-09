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

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        return F.tanh(x)
