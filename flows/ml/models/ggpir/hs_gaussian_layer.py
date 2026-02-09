from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from flows.ml.layers.sep_gaussian import SepGaussianLayer
from .hs_gaussian import HSGaussian



class HSGaussianLayer(SepGaussianLayer):

    def create_gaussian(
        self,
        x_channels: int,
        y_channels: int,
        **kwargs,
    ) -> HSGaussian:
        return HSGaussian(x_channels, y_channels)