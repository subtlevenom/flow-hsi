from typing import List, Union
import torch
from einops import rearrange, einsum


class Rearrange(torch.nn.Module):

    def __init__(
        self,
        pattern: str,
        **axes_lengths,
    ):
        super(Rearrange, self).__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths or {}

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        y = rearrange(
            x,
            self.pattern,
            **self.axes_lengths,
        )
        return y