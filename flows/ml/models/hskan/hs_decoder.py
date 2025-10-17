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
        self.mid_channels = in_channels * out_channels

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=self.mid_channels,
                kernel_size=1,
            ), nn.ReLU(),
            nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=out_channels,
                kernel_size=1,
            ))

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape
        x = rearrange(
            x,
            '(b i) o h w -> b (i o) h w',
            i=self.out_channels,
            o=self.in_channels,
        )
        x = F.sigmoid(self.proj(x))
        return x
