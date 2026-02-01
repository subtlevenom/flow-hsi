import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flows.ml.layers.encoders import CMEncoder, LightCMEncoder
from flows.ml.layers.encoders.sg_encoder import FFN, LayerNorm
from flows.ml.models.ggpd.gaussian import Gaussian
from .utils import covariance_matrix


class GPEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 30,
    ):
        super(GPEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = CMEncoder(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        I = self.in_channels
        C = 2 * I

        p = self.encoder(x)
        m = p[:, :C]
        s = p[:, C:2*C]
        s = torch.square(F.relu(s) - F.relu(-s)) 
        s = torch.clip(s,1e-6)
        a = 2 * torch.pi * F.sigmoid(p[:, 2*C:-I])
        x = p[:,-I:]
        S,R,D = covariance_matrix(s, a)

        # S^-1 = (CT*D*C)^-1 = C*D^-1*CT
        D_1 = torch.linalg.inv(D)
        RT = R.permute(0,1,2,4,3)
        S_1 = torch.matmul(R,D_1)
        S_1 = torch.matmul(S_1,RT)

        # y = my + Syx * Sxx^-1 * (x-mx)
        Sxx = S_1[:,:,:,:I,:I]
        Syy = S_1[:,:,:,I:,I:]
        Syx = S_1[:,:,:,I:,:I]
        Sxy = S_1[:,:,:,:I,I:]
        Sxx_1 = torch.linalg.inv(Sxx)
        Q = torch.matmul(Syx,Sxx_1)
        mx = m[:,:I]
        my = m[:,I:]
        y = my + torch.einsum('bijmn,bnij->bmij', Q, (x-mx))

        # to use correctly in ggpd_loss
        S = S.permute(0,3,4,1,2)

        return x, m, S, y
