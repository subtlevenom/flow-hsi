from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flows.ml.layers.sep_kan import SepKAN
from flows.ml.layers.encoders import HSKANEncoder


class HSKAN(SepKAN):

    def create_encoder(self, in_channels, out_channels, **kwargs):
        return HSKANEncoder(in_channels=31, out_channels=out_channels)

    def forward(self, x, w):
        w = self.encoder(w)
        w = w.repeat_interleave(x.shape[0]//w.shape[0], dim=1)
        x = self.sep_kan_layer(x,w)
        return x
