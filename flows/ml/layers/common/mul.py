from typing import List
from functools import reduce
from operator import mul
import torch


class Mul(torch.nn.Module):

    def __init__( self, **kwargs):
        super(Mul, self).__init__()

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        x = reduce(mul, x)
        return x
