from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..feed import SepConvAtt


class FusionAtt(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        activation: int = nn.Sigmoid,
        **kwargs: Any,
    ):
        super(FusionAtt, self).__init__()

        self.feed = SepConvAtt(in_channels=sum(in_channels),
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               activation=activation)

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(x, dim=1)
        x = self.feed(x)
        return x
