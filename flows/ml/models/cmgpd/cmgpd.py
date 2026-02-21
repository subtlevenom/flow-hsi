from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flows.ml.layers.sep_gpd import SepGPD, SepGPDLayer
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder


class CmGPDLayer(SepGPDLayer):

    def create_encoder(self, in_channels, out_channels, **kwargs):
        return CMEncoder(in_channels, out_channels)


class LightCmGPDLayer(SepGPDLayer):

    def create_encoder(self, in_channels, out_channels, **kwargs):
        return LightCMEncoder(in_channels, out_channels)


class CmGPD(SepGPD):

    def create_layer(self, in_channels: int, out_channels: int,
                     s_range: List[int], **kwargs) -> SepGPDLayer:
        return CmGPDLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            s_range=s_range,
        )


class LightCmGPD(SepGPD):

    def create_layer(self, in_channels: int, out_channels: int,
                     s_range: List[int], **kwargs) -> SepGPDLayer:
        return LightCmGPDLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            s_range=s_range,
        )
