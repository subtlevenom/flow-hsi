import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.models.ggpd.gaussian import Gaussian
from .utils import covariance_matrix


class GPEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 30,
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = CMEncoder(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        p = self.encoder(x)
        C = 2 * self.in_channels
        m = p[:, :C]
        s = p[:, C:2*C]
        s = torch.square(F.relu(s) - F.relu(-s)) 
        a = 2 * torch.pi * F.sigmoid(p[:, 2*C:-self.in_channels])
        x = p[:,-self.in_channels:]
        r,c = covariance_matrix(s, a)

        y = torch.cat([x, m[:,self.in_channels:]], dim=1)
        y = torch.einsum('bcij,bclij->blij', s*(y-m), c)
        y = m[:,self.in_channels:] + (1. / s[:,3:]) * y[:,self.in_channels:]

        return x, m, r, y
