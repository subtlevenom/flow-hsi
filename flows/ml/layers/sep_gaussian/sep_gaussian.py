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
        self.encoder = self.create_encoder(in_channels, in_channels + 1 + 6)

    @abstractmethod
    def create_encoder(self, in_channels:int, out_channels:int, **kwargs):
        return NotImplemented

    def forward(self, x:torch.Tensor):
        B,C,H,W = x.shape
        C = C-1
        w = self.encoder(x)
        c = w[:,:6]
        a = w[:,6:7]
        m = w[:,7:C+7]
        s = w[:,C+7:]
        y = self.gaussian(x,c,a,m,s)
        return y
