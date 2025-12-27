from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from .gaussian import Gaussian


class SepGaussian(ABC, torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
    ):
        super(SepGaussian, self).__init__()

        self.in_channels = in_channels
        self.gaussian = Gaussian()
        self.encoder = self.create_encoder(in_channels, (in_channels + 1) * in_channels)

    @abstractmethod
    def create_encoder(self, in_channels:int, out_channels:int, **kwargs):
        return NotImplemented

    def forward(self, x:torch.Tensor):
        B,C,H,W = x.shape
        w = self.encoder(x) # B (C+1)*C H W
        m = w[:,:C]
        s = w[:,C:].view(B,C,C,H,W)
        y = self.gaussian(x,m,s)
        return y
