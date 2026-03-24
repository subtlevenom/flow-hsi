from typing import List
import torch


class Sum(torch.nn.Module):

    def __init__( self, **kwargs):
        super(Sum, self).__init__()

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        x = sum(x)
        return x
