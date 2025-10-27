import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HSEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 31,
        out_channels: int = 3,
    ):
        super(HSEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = in_channels * out_channels

        self.weight = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=1,
            ),
        )

        self.value = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
        )

        self.proj = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape

        a = rearrange(x, 'b c h w -> b (h w) c')
        U,S,V = torch.pca_lowrank(a, q=4, center=False, niter=2)
        U = rearrange(U, 'b (h w) c -> b c h w', h=H, w=W)

        return U,S,V
