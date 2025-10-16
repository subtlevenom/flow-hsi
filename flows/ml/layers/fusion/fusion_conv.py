from typing import Any, List
import torch
import torch.nn as nn
from ..feed import ConvBlock


class FusionConv(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_sizes: List[int] = [3, 3],
        activation: int = nn.ReLU,
        **kwargs: Any,
    ):
        super(FusionConv, self).__init__()

        in_channels=sum(in_channels)

        self.feed = ConvBlock(in_channels=sum(in_channels),
                              out_channels=out_channels,
                              kernel_sizes=kernel_sizes,
                              activation=activation)

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(x, dim=1)
        x = self.feed(x)
        return x
