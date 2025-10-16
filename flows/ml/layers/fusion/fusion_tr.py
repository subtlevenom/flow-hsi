from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from einops import rearrange
from ..transformer import TransformerPredictor
from ..transformer.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from ..transformer.position_encoding import PositionEmbeddingSine


class FusionTR(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_dim: int = 7,
        nheads: int = 8,
        nlayers: int = 6,
        dim_feedforward: int = 1024,
        normalize_before=False,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super(FusionTR, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_dim = feature_dim
        self.dropout = dropout

        feature_dim = feature_dim * feature_dim

        # decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=in_channels,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            normalize_before=False)
        encoder_norm = nn.LayerNorm(in_channels) if normalize_before else None
        self.prob_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=nlayers,
            norm=encoder_norm,
        )
        self.pos_embedding = nn.Embedding(feature_dim, in_channels)

        self.feed = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> (h w) b c')
        y = rearrange(y, 'b c h w -> (h w) b c')

        pos_embed = self.pos_embedding.weight.unsqueeze(1).repeat(1, B, 1)
        x = self.prob_decoder(x, y, pos=pos_embed, query_pos=pos_embed)

        x = rearrange(x[0], '(h w) b c -> b c h w', h = H, w = W)
        x = self.feed(x)

        return x
