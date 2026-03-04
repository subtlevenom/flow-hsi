from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flows.ml.layers.sep_gpd import SepGPD, SepGPDLayer
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.mst import MSABProjector, MSABCMProjector, MST, MST_Plus_Plus


class GPDLayer(SepGPDLayer):

    def create_encoder(self,
                       in_channels: int,
                       out_channels: int,
                       alg: str = 'mst',
                       **kwargs):
        """
        kwargs['alg']: 'msab','cmlight','cm','mix',None
        kwargs['num_blocks']: [2,2]
        """

        match alg:
            case 'mst':
                return MST(
                    in_dim=in_channels,
                    out_dim=out_channels,
                    dim=in_channels,
                    num_blocks=kwargs.get('num_blocks', [2, 4, 4]),
                )
            case 'mst++':
                return MST_Plus_Plus(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_feat=in_channels,
                )
            case 'msab':
                return MSABProjector(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_blocks=kwargs.get('num_blocks', [2, 2]),
                )
            case 'lightcm':
                return LightCMEncoder(in_channels, out_channels)
            case 'cm':
                return CMEncoder(in_channels, out_channels)
            case 'mix':
                return MSABCMProjector(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_blocks=kwargs.get('num_blocks', [2, 2]),
                )
            case _:
                return None


class GPD(SepGPD):

    def create_layer(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> SepGPDLayer:
        return GPDLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
