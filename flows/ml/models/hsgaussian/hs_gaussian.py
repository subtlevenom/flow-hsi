from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flows.ml.layers.sep_gaussian import SepGaussian
from flows.ml.layers.encoders import SGEncoder, CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MST_Plus_Plus


class HSGaussian(SepGaussian):

    def create_encoder(self, in_channels, out_channels, **kwargs):
        return MST_Plus_Plus(
            in_channels=in_channels,
            out_channels=out_channels,
            n_feat=20,
            stage=2
        )
        return LightCMEncoder(in_channels, out_channels)

    def forward(self, x):
        y = super(HSGaussian, self).forward(x)
        return y