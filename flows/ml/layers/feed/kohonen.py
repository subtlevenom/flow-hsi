from typing import Any, List
from einops import einsum, rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kohonen(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """

    def __init__(self, in_channels:int, out_channels:int, num_slots:int = None):
        super().__init__()

        num_slots = num_slots or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_slots = num_slots

        self.q = nn.Conv2d(in_channels, num_slots, kernel_size=1)
        self.k = nn.Conv2d(in_channels, num_slots, kernel_size=1)
        self.v = nn.Parameter(torch.rand(num_slots, out_channels))

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape

        # basis coeff
        q = self.q(x)
        k = self.k(x)

        p = F.softmax(q*k, dim=1)

        y = torch.einsum('bshw,sc -> bchw', p, self.v)

        return y