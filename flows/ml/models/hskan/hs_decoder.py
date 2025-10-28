import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HSDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 31,
    ):
        super(HSDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor):

        B, C, H, W = U.shape

        U = rearrange(U, 'b c h w -> b (h w) c')
        U = U * S.unsqueeze(1)
        A = torch.bmm(U, V.permute(0,2,1))
        x = rearrange(A, 'b (h w) c -> b c h w', h=H,w=W)

        return x


