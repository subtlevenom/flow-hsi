from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import einops
from .multivariate_normal import MultivariateNormal


class SepGPDLayer(ABC, torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        s_range: List[float] = [1e-3, 1e+3],
    ):
        super(SepGPDLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        m_channels = out_channels
        s_channels = out_channels
        a_channels = s_channels * (s_channels - 1) // 2

        self.m_channels = m_channels
        self.s_channels = s_channels
        self.a_channels = a_channels

        s_range = s_range or [1e-3, 1e+3]
        self.smin = s_range[0]
        self.smax = s_range[1]

        self.encoder = self.create_encoder(
            in_channels, m_channels + s_channels + a_channels)

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

        # D
        D = torch.diag_embed(s.permute(0, 2, 3, 1))
        # R
        RT = R.transpose(3, 4)
        Q = torch.einsum('bmnij,bmnjk -> bmnik', RT, D)
        # RT*D*R
        S = torch.einsum('bmnij,bmnjk -> bmnik', Q, R)

        return S, R, D

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        w = self.encoder(x)

        m = w[:, :self.m_channels].permute(0, 2, 3, 1)
        s = F.sigmoid(w[:, self.m_channels:self.m_channels + self.s_channels])
        s = self.smin * (1 - s) + self.smax * s
        a = torch.pi * F.tanh(w[:, self.m_channels + self.s_channels:])

        S, _, _ = self.covariance_matrix(s, a)
        S = 0.5 * (S + S.transpose(3, 4))  # to minimize rounding errors

        return MultivariateNormal(mean=m, covariance_matrix=S)
