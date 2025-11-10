import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PCADecoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 31,
    ):
        super(PCADecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        s: torch.Tensor,
        v: torch.Tensor,
    ):
        B, C, H, W = u.shape

        u = rearrange(u, 'b c h w -> b (h w) c')
        u = u * s.unsqueeze(1)
        a = torch.bmm(u, v.permute(0,2,1))
        y = rearrange(a, 'b (h w) c -> b c h w', h=H,w=W)

        return x + y
