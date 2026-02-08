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

        self.layer = GPGaussian(x_channels=in_channels, y_channels=out_channels)

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

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        m,S,R,D = self.layer(x)

        y_ = self.get_my_by_x(x,m,R,D)

        RT = R.permute(0,1,2,4,3)
        z_ = torch.cat([x, y_], dim=1) - m
        z_ = torch.einsum('bmij,bijmn->bnij', z_, RT)
        z_[:,:3] = -z_[:,:3]
        z_ = torch.einsum('bmij,bijmn->bnij', z_, R) + m

        x_ = z_[:,:3]
        y_ = z_[:,3:]

        S = torch.permute(S, (0, 3, 4, 1, 2))

        return x_, y_, m, S
