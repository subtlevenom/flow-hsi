from typing import Any
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        **kwargs: Any,
    ):
        super(SimpleHead, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              1,
                              kernel_size=1,
                              stride=1,
                              padding=0)

        self.feed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1),
        )

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape
        y = self.conv(x)
        x = self.feed(x * y)
        return x.view((B, -1))
