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
from flows.ml.layers.transformer import TransformerEncoder, TransformerEncoderLayer


class DETREncoder(nn.Module):

    def __init__(
        self,
        backbone: str = 'efficientnet-b2',
        in_channels: int = 3,
        out_channels: int = 489,
        feature_dim: int = 7,
        nheads: int = 8,
        nlayers: int = 6,
        dim_feedforward: int = 1024,
        normalize_before=False,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super(DETREncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_dim = feature_dim

        feature_dim = feature_dim * feature_dim

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

        encoder_layer = TransformerEncoderLayer(
            d_model=out_channels,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            normalize_before=False)
        encoder_norm = nn.LayerNorm(in_channels) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=nlayers,
                                          norm=encoder_norm)

        self.pos_embedding = nn.Embedding(feature_dim + 1, out_channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))

        if out_channels == hidden_channels:
            self.out_proj = nn.Sequential()
        else:
            self.out_proj = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = self.in_proj(x)

        x = self.backbone(x)[-1]
        x = self.out_proj(x)

        x = x.flatten(2).permute(2, 0, 1)

        pos_embed = self.pos_embedding.weight.unsqueeze(1).repeat(1, B, 1)
        cls_embed = self.cls_token.repeat((1, B, 1))

        x = torch.cat([cls_embed, x], 0)
        x = self.encoder(x, src_key_padding_mask=None, pos=pos_embed)

        x = x.permute(1, 2, 0)

        return x
