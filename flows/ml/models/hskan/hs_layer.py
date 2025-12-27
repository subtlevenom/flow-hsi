import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..hsgaussian import HSGaussian
from flows.ml.layers.encoders.sg_encoder import FFN


class HSLayer(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_gaussian: int = 11,
    ):
        super(HSLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.ModuleList(
            [HSGaussian(in_channels) for _ in range(n_gaussian)])

        n_channels = n_gaussian + in_channels
        self.ffn = FFN(in_channels=n_channels, out_channels=n_channels)
        self.out = nn.Conv2d(in_channels=n_channels,
                             out_channels=out_channels,
                             kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = torch.concat([g(x) for g in self.layers] + [x], dim=1)
        w = self.ffn(y)
        # y = torch.softmax(w, dim=1) * y
        y = self.out(w * y)
        return y
