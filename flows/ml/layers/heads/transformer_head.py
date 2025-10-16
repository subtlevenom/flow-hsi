from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class TRHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: List[int] = [3, 2],
        feature_dim: int = 7,
        nheads: int = 8,
        nlayers: int = 6,
        dim_feedforward: int = 1024,
        normalize_before=False,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super(TRHead, self).__init__()

        feature_dim = feature_dim * feature_dim

        # encoder
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

        # decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=in_channels,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            normalize_before=False)
        decoder_norm = nn.LayerNorm(in_channels) if normalize_before else None
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=nlayers,
            norm=decoder_norm,
        )

        # embeddings
        self.k_pos_embedding = nn.Embedding(feature_dim + 1, in_channels)
        self.q_pos_embedding = nn.Embedding(feature_dim + 1, in_channels)

        self.k_token = nn.Parameter(torch.randn(1, 1, in_channels))
        self.q_token = nn.Parameter(torch.randn(1, 1, in_channels))

        # feedforward
        self.k_feed = nn.Linear(in_channels, out_channels[0])
        self.q_feed = nn.Linear(in_channels, out_channels[1])

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x - key, y - query

        B, C, H, W = x.shape

        x = x.flatten(2).permute(2, 0, 1)
        y = y.flatten(2).permute(2, 0, 1)

        k_pos_embed = self.k_pos_embedding.weight.unsqueeze(1).repeat(1, B, 1)
        q_pos_embed = self.q_pos_embedding.weight.unsqueeze(1).repeat(1, B, 1)

        k_token = self.k_token.repeat((1, B, 1))
        q_token = self.q_token.repeat((1, B, 1))

        x = torch.cat([k_token, x], 0)
        y = torch.cat([q_token, y], 0)

        x = self.encoder(x, src_key_padding_mask=None, pos=k_pos_embed)
        y = self.encoder(y, src_key_padding_mask=None, pos=q_pos_embed)

        xx = self.k_feed(x[0])
        yy = self.q_feed(y[0])

        return xx, yy
