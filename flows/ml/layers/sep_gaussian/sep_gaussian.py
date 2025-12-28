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
        self.encoder = self.create_encoder(in_channels, 2 * in_channels + 1)

    @abstractmethod
    def create_encoder(self, in_channels:int, out_channels:int, **kwargs):
        return NotImplemented

    def forward(self, x:torch.Tensor):
        B,C,H,W = x.shape
        w = self.encoder(x)
        a = w[:,:1]
        m = w[:,1:C+1]
        s = w[:,C+1:]
        y = self.gaussian(x,a,m,s)
        return y
