import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder


class GPCorrector(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super(GPCorrector, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj = LightCMEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor):
        return self.proj(x)
