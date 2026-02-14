from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from .gaussian import Gaussian


class GPDGaussian(ABC, torch.nn.Module):

    def __init__(
        self,
        x_channels: int,
        y_channels: int,
    ):
        super(GPDGaussian, self).__init__()

        self.x_channels = x_channels
        self.y_channels = y_channels

        m_channels = x_channels + y_channels
        s_channels = x_channels + y_channels
        a_channels = s_channels*(s_channels-1) // 2

        self.m_channels = m_channels
        self.s_channels = s_channels
        self.a_channels = a_channels
        
        self.smax = 1e+2

        self.encoder = self.create_encoder(
            x_channels, m_channels + s_channels + a_channels)

    @abstractmethod
    def create_encoder(self, in_channels: int, out_channels: int, **kwargs):
        return NotImplemented

    def covariance_matrix(self, s:torch.Tensor, a:torch.Tensor):
        B,C,H,W = s.shape
        R = None

        cs = torch.cos(a)
        ss = torch.sin(a)

        c = 0
        for i in range(C-1):
            for j in range(i+1,C):
                rij = torch.zeros((B,H,W,C,C),dtype=a.dtype, device=a.device)
                rij[:,:,:] = torch.eye(C)
                cij = cs[:,c]
                sij = ss[:,c]
                rij[:,:,:,i,i] = cij
                rij[:,:,:,j,j] = cij
                rij[:,:,:,i,j] = -sij
                rij[:,:,:,j,i] = sij
                R = rij if c == 0 else torch.matmul(R, rij)
                c += 1

        # C
        RT = R.permute(0,1,2,4,3)
        # D
        D = torch.zeros((B,H,W,C,C),dtype=a.dtype, device=a.device)
        D[:,:,:] = torch.eye(C)
        D = D * torch.square(1 / s).unsqueeze(2).permute(0,3,4,1,2)
        # RT*D*R
        S = torch.matmul(RT,D)
        S = torch.matmul(S,R)

        return S,R,D

    def forward(self, x: torch.Tensor):
        w = self.encoder(x)

        m = w[:,:self.m_channels]
        s = 1./self.smax + self.smax * F.sigmoid(w[:,self.m_channels:self.m_channels+self.s_channels])
        a = torch.pi * F.tanh(w[:,self.m_channels+self.s_channels:])

        S,R,D = self.covariance_matrix(s, a)

        return m, S, R, D, s
