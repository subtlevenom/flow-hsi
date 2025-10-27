from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        return self.linear(x)
