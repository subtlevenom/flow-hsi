from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
from torchvision.transforms import Normalize
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp


class SmpEncoder(nn.Module):

    def __init__(
        self,
        arch: str = 'unet',
        backbone: str = 'efficientnet-b2',
        in_channels: int = 3,
        out_channels: int = 3,
        feat_channels: int = 289,
        layers: int = 5,
        activation: str = 'sigmoid',
        features_only: bool = False,
        **kwargs,
    ):
        super(SmpEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.features_only = features_only

        self.model = smp.create_model(
            arch=arch,
            encoder_name=backbone,  # 'efficientnet-b4', 'resnet18',
            encoder_depth=layers,
            encoder_weights='imagenet',
            activation=activation,
            in_channels=in_channels,
            classes=out_channels,
        )

        hidden_channels = self.model.encoder.out_channels[-1]

        if feat_channels == hidden_channels:
            self.out_proj = nn.Sequential()
        else:
            self.out_proj = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.model.encoder(x)
        return x

    def decode(self, x: List[torch.Tensor]) -> torch.Tensor:
        x = self.model.decoder(x)
        x = self.model.segmentation_head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        f = self.encode(x)
        x = self.out_proj(f[-1])

        if self.features_only:
            return x

        y = self.decode(f)
        return x, y
