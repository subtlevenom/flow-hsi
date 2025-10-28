from typing import Any
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transformer import TransformerEncoder, TransformerEncoderLayer


class Basis(nn.Module):

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 31,
        nheads: int = 8,
        nlayers: int = 6,
        dim_feedforward: int = 1024,
        normalize_before=False,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super(Basis, self).__init__()

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
        self.pos_embedding = nn.Embedding(out_channels, in_channels)

    def forward(self, x: torch.Tensor):

        B, C, N = x.shape
        x = x.permute(1, 0, 2) # C B N

        pos_embed = self.pos_embedding.weight.unsqueeze(1).repeat(1, B, 1)
        x = self.encoder(x, src_key_padding_mask=None, pos=pos_embed)
        x = x.permute(1, 0, 2) # B C N

        return x
