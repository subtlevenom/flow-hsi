from typing import Any
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transformer import TransformerEncoder, TransformerEncoderLayer


class DETRHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        feature_dim: int = 7,
        nheads: int = 8,
        nlayers: int = 6,
        dim_feedforward: int = 1024,
        normalize_before=False,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super(DETRHead, self).__init__()

        feature_dim = feature_dim * feature_dim

        encoder_layer = TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(in_channels) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=nlayers,
                                          norm=encoder_norm)

        self.pos_embedding = nn.Embedding(feature_dim + 1, in_channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channels))

        self.feed = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)

        pos_embed = self.pos_embedding.weight.unsqueeze(1).repeat(1, B, 1)
        cls_token = self.cls_token.repeat((1, B, 1))

        x = torch.cat([cls_token, x], 0)

        x = self.encoder(x, src_key_padding_mask=None, pos=pos_embed)

        x = self.feed(x[0])

        return x
