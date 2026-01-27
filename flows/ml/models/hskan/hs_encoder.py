import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm


class HSEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_iter: int = 0
    ):
        super(HSEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if n_iter == 0:
            self.proj = FFN(in_channels=in_channels, out_channels=out_channels)
        else:
            from .hs_net import HSNet
            self.proj = HSNet(in_channels, out_channels, n_iters=n_iter-1)

    def forward(self, x: torch.Tensor):
        return self.proj(x)
