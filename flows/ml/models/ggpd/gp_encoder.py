import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
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
        n_layers:int = 3
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.ModuleList([
            GPGaussian(x_channels=in_channels, y_channels=out_channels)
            for _ in range(n_layers)
        ])
        self.g = Gaussian()

    def get_my_by_x(self,x,m,R,D):
        C = self.in_channels

        # S^-1 = (RT*D*R)^-1 = R*D^-1*CT
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
        y = my + torch.einsum('bijmn,bnij->bmij', Q, (x-mx))

        return y

    def get_mx_by_y(self,y,m,R,D):
        C = self.out_channels

        # S^-1 = (RT*D*R)^-1 = R*D^-1*CT
        D_1 = torch.linalg.inv(D)
        RT = R.permute(0,1,2,4,3)
        S_1 = torch.matmul(R,D_1)
        S_1 = torch.matmul(S_1,RT)

        # y = my + Syx * Sxx^-1 * (x-mx)
        Syy = S_1[:,:,:,C:,C:]
        Sxy = S_1[:,:,:,:C,C:]
        Syy_1 = torch.linalg.inv(Syy)
        Q = torch.matmul(Sxy,Syy_1)
        mx = m[:,:C]
        my = m[:,C:]
        x = mx + torch.einsum('bijmn,bnij->bmij', Q, (y-my))

        return x

    def forward(self, x: torch.Tensor, tgt: torch.Tensor):

        _x = []
        _y = []
        _m = []
        _S =[]
        _p = []
        _d = []

        for layer in self.layers:
            xx,m,S,R,D = layer(x)
            S = S.permute(0,3,4,1,2)
            yy = self.get_my_by_x(xx,m,R,D)
            z = torch.cat([xx,yy], dim=1)
            p = self.g(z, m, S)
            y0 = self.get_my_by_x(x,m,R,D)
            z = torch.cat([x,y0], dim=1)
            q = self.g(z, m, S)
            # p = q / p
            # p = 2 * (q / (w + q) - 0.5)
            # y_sum += m[:,3:] * p
            # p_sum += p
            _x.append(xx)
            _y.append(yy)
            _m.append(m)
            _S.append(S)
            _p.append(p)
        # y = y_sum / p_sum

        return _x, _y, _m, _S, _p
