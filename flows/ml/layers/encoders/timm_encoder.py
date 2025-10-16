from typing import Any, List
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
from torchvision.transforms import Normalize
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp


class TimmEncoder(nn.Module):

    def __init__(
        self,
        backbone: str = 'efficientnet-b2',
        in_channels: int = 3,
        out_channels: int = 256,
        **kwargs: Any,
    ):
        super(TimmEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == 3:
            self.in_proj = nn.Sequential()
        else:
            self.in_proj = nn.Conv2d(in_channels=in_channels,
                                     out_channels=3,
                                     kernel_size=1)

        self.backbone = timm.create_model(backbone,
                                          features_only=True,
                                          pretrained=True)

        hidden_channels = self.backbone.feature_info.channels()[-1]

        if out_channels == hidden_channels:
            self.out_proj = nn.Sequential()
        else:
            self.out_proj = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.backbone(x)[-1]
        x = self.out_proj(x)
        return x
