from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from .multivariate_normal import MultivariateNormal


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
        a_channels = s_channels * (s_channels - 1) // 2

        self.m_channels = m_channels
        self.s_channels = s_channels
        self.a_channels = a_channels

        self.smin = 1e-3
        self.smax = 1e+3

        self.encoder = self.create_encoder(
            x_channels, m_channels + s_channels + a_channels)

    @abstractmethod
    def create_encoder(self, in_channels: int, out_channels: int, **kwargs):
        return NotImplemented

    def covariance_matrix(self, s: torch.Tensor, a: torch.Tensor):
        B, C, H, W = s.shape
        R = None

        cs = torch.cos(a)
        ss = torch.sin(a)

        c = 0
        for i in range(C - 1):
            for j in range(i + 1, C):
                rij = torch.zeros((B, H, W, C, C),
                                  dtype=a.dtype,
                                  device=a.device)
                rij[:, :, :] = torch.eye(C)
                cij = cs[:, c]
                sij = ss[:, c]
                rij[:, :, :, i, i] = cij
                rij[:, :, :, j, j] = cij
                rij[:, :, :, i, j] = -sij
                rij[:, :, :, j, i] = sij
                R = rij if c == 0 else torch.matmul(R, rij)
                c += 1

        # R
        RT = R.transpose(3,4)
        # D
        D = torch.diag_embed(s.permute(0, 2, 3, 1))
        # RT*D*R
        S = RT @ D @ R

        return S, R, D

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        w = self.encoder(x)

        m = w[:, :self.m_channels].permute(0,2,3,1)

        s = w[:, self.m_channels:self.m_channels + self.s_channels]
        s = F.sigmoid(s)
        s = self.smin * (1 - s) + self.smax * s

        a = w[:, self.m_channels + self.s_channels:]
        a = torch.pi * F.tanh(a)

        S, _, _ = self.covariance_matrix(s, a)

        return MultivariateNormal(mean=m, covariance_matrix=S), s
