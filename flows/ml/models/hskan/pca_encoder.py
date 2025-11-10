import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PCAEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 31,
        out_channels: int = 3,
    ):
        super(PCAEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        u, s, v = torch.pca_lowrank(x,
                                    q=self.out_channels,
                                    center=False,
                                    niter=2)
        u = rearrange(u, 'b (h w) c -> b c h w', h=H, w=W)

        return u, s, v
