from typing import List
import torch


class Const(torch.nn.Module):

    def __init__( self, value: int, **kwargs):
        super(Const, self).__init__()
        self.value = value

    def forward(self) -> torch.Tensor:
        return self.value
