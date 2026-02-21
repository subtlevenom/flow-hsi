import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Sink(nn.Module):

    def __init__(
        self,
        height: int = 256,
        width: int = 256,
    ):
        super(Sink, self).__init__()

        self.height = height
        self.width = width

        self.a = nn.Parameter(torch.ones(1, 1, height, width))
        self.b = nn.Parameter(torch.zeros(1, 1, height, width))

    def forward(self, x: torch.Tensor):

        x = self.a * x + self.b

        return x
