import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HSDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 31,
    ):
        super(HSDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        return x + self.proj(y)
