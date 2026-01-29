import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from .utils import covariance_matrix


class GPEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 27,
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = CMEncoder(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        m = x[:, :2 * self.in_channels]
        s = covariance_matrix(x[:, 2 * self.in_channels:])
        return m, s
