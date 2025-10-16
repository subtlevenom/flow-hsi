import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from flows.ml.layers.kan import SepKANLayer2D


class HSKANEncoder(nn.Module):

    def __init__(self,
                 in_channels: int = 31,
                 out_channels: int = 50,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 residual_std: float = 0.1,
                 grid_range: list = [0, 1]):
        super(HSKANEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = in_channels+5

        self.a0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
            )
        self.a1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
            )
        self.a2 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
            )
        self.a3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
            )
        self.a4 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
            )

        self.b1 = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                groups=mid_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                groups=mid_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                groups=mid_channels // 4,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                groups=mid_channels // 6,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )

        self.w = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        self.v = nn.Conv2d(
                in_channels=5,
                out_channels=3 * 16,
                kernel_size=1,
            )
        self.shuffle = nn.PixelShuffle(4)

    def forward(self, x: torch.Tensor):

        x0 = self.a0(x)
        x1 = self.a1(x)
        x2 = self.a2(x)
        x3 = self.a3(x)
        x4 = self.a4(x)

        y0 = torch.cat([x0, x[:,0:8], x1, x[:,8:15], x2, x[:,15:23], x3, x[:,23:], x4], dim=1)
        y1 = self.b1(y0)
        y2 = self.b2(y1 + y0)
        y3 = self.b3(y2 + y1)
        y4 = self.b4(y3 + y2)

        y = y4
        y = torch.cat([y[:,1:9], y[:,10:17], y[:,18:26], y[:,27:35]], dim=1)
        w:torch.Tensor = self.w(y)
        w = w.repeat_interleave(repeats=4, dim=-1)
        w = w.repeat_interleave(repeats=4, dim=-2)

        z = y4
        z = torch.cat([z[:,0:1], z[:,8:9], z[:,17:18], z[:,26:27], z[:,35:]], dim=1)
        v = F.sigmoid(self.v(z))
        v = self.shuffle(v)

        return v, w
