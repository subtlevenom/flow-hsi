from typing import List
import torch


class Concat(torch.nn.Module):

    def forward(self, *x: List[torch.Tensor], dim:int=1) -> torch.Tensor:
        x = torch.cat(x, dim=dim)
        return x
