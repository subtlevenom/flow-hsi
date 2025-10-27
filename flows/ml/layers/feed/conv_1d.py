from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size:int=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        return self.conv(x)
