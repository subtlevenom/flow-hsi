from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flows.ml.layers.gpd_gaussian import GPDGaussian
from flows.ml.layers.encoders import SGEncoder, CMEncoder, LightCMEncoder


class GPGaussian(GPDGaussian):

    def create_encoder(self, in_channels, out_channels, **kwargs):
        return LightCMEncoder(in_channels, out_channels)