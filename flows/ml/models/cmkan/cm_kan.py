from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flows.ml.layers.sep_kan import SepKAN
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder


class CmKAN(SepKAN):

    def create_encoder(self, in_channels, out_channels, **kwargs):
        return CMEncoder(in_channels, out_channels)

    def forward(self, x):
        y = super(CmKAN, self).forward(x)
        return y - x
        # y = super(CmKAN, self).forward(x[:,:self.in_channels])
        # x = torch.cat([y,x[:,self.out_channels:]], dim=1)
        # return x


class LightCmKAN(CmKAN):

    def create_encoder(self, in_channels, out_channels, **kwargs):
        return LightCMEncoder(in_channels, out_channels)
