import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.layers.gpd_gaussian import Gaussian
from .gp_gaussian import GPGaussian 


class GPEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gaussian = Gaussian()

        self.layers = nn.ModuleList(
            [GPGaussian(x_channels=in_channels, y_channels=1) for _ in range(out_channels)])

    def forward(self, x: torch.Tensor):

        C = self.in_channels

        x_channels = []
        y_channels = []
        p_channels = []

        for layer in self.layers:
            x_,m,S,R,D = layer(x)

            # S^-1 = (CT*D*C)^-1 = C*D^-1*CT
            D_1 = torch.linalg.inv(D)
            RT = R.permute(0,1,2,4,3)
            S_1 = torch.matmul(R,D_1)
            S_1 = torch.matmul(S_1,RT)

            # y = my + Syx * Sxx^-1 * (x-mx)
            Sxx = S_1[:,:,:,:C,:C]
            Syx = S_1[:,:,:,C:,:C]
            Sxx_1 = torch.linalg.inv(Sxx)
            Q = torch.matmul(Syx,Sxx_1)
            mx = m[:,:C]
            my = m[:,C:]
            y_ = my + torch.einsum('bijmn,bnij->bmij', Q, (x-mx))

            x_channels.append(x_)
            y_channels.append(y_)

            S = torch.permute(S, (0, 3, 4, 1, 2))
            yx = torch.cat([x_, y_], dim=1)
            p = self.gaussian(yx, m, S)
            p_channels.append(p)

        x = torch.cat(x_channels, dim=1)
        y = torch.cat(y_channels, dim=1)

        p = torch.cat(p_channels, dim=1)
        p = torch.prod(p, dim=1, keepdim=True)

        return x, y, p
