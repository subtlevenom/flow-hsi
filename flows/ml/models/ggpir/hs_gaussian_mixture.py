from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from flows.ml.layers.sep_gaussian import SepGaussianMixture
from .hs_gaussian_layer import HSGaussianLayer


class HSGaussianMixture(SepGaussianMixture):

    def create_gaussian_layer(
        self,
        x_channels: int,
        y_channels: int,
        g_channels: int,
        **kwargs,
    ) -> HSGaussianLayer:
        return HSGaussianLayer(x_channels, y_channels, g_channels)
