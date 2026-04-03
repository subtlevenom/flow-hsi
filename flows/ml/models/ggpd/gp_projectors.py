from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, einsum
from .gp_projector import GPProjector


class GPProjectors(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        alg: str = 'mix',
        num_blocks: List[int] = [2, 2],
        **kwargs,
    ):
        super(GPProjectors, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = 2 * in_channels + 1
        self.out_channels = out_channels

        self.index = torch.arange(self.mid_channels).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        self.projectors = nn.ModuleList([
            GPProjector(
                in_channels=in_channels,
                out_channels=self.mid_channels,
                alg=alg,
                num_blocks=num_blocks,
                **kwargs,
            ) for _ in range(out_channels)
        ])

    def forward(self, x: torch.Tensor):
        """
        BCHW -> NB(2C+1)HW
        C - in_channels
        N - out_channels
        """
        s = self.index.to(x.device)
        n = self.mid_channels - 1

        y = []
        for projector in self.projectors:
            _y = projector(x)
            _y = torch.cumsum(F.sigmoid(_y), dim=1)
            y.append((_y + s) / n)
        y = torch.stack(y,dim=0) # nbchw
        return y
