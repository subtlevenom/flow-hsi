from typing import List
import torch


class Concat(torch.nn.Module):

    def __init__( self, dim: int = 1, **kwargs):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *x: List[torch.Tensor], dim:int=None) -> torch.Tensor:
        dim = self.dim if dim is None else dim
        x = torch.cat(x, dim=dim)
        return x
