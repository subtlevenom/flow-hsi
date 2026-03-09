from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import einops
from .multivariate_normal import MultivariateNormal


class SepGPDLayer(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        sigma_range: List[int] = [1e-3, 1e+3],
        **kwargs,
    ):
        super(SepGPDLayer, self).__init__()

        self.dim = dim

        m_channels = dim
        s_channels = dim
        a_channels = m_channels * (m_channels - 1) // 2

        self.m_channels = m_channels
        self.s_channels = s_channels
        self.a_channels = a_channels

        sigma_range = kwargs.get('sigma_range', [1e-3, 1e+3])
        self.sigma_min = sigma_range[0]
        self.sigma_max = sigma_range[1]

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

        if R is None:
            R = torch.ones((B,H,W,1,1)).to(s.device)

        # D
        D = torch.diag_embed(s.permute(0, 2, 3, 1))
        # R
        RT = R.transpose(3, 4)
        Q = torch.einsum('bmnij,bmnjk -> bmnik', RT, D)
        # RT*D*R
        S = torch.einsum('bmnij,bmnjk -> bmnik', Q, R)

        return S, R, D

    def forward(self, x: torch.Tensor) -> MultivariateNormal:

        m = x[:, :self.m_channels].permute(0, 2, 3, 1)
        s = F.sigmoid(x[:, self.m_channels:self.m_channels + self.s_channels])
        s = self.sigma_min * (1 - s) + self.sigma_max * s
        a = torch.pi * F.tanh(x[:, self.m_channels + self.s_channels:])

        S, _, _ = self.covariance_matrix(s, a)
        S = 0.5 * (S + S.transpose(3, 4))  # to minimize rounding errors

        return MultivariateNormal(mean=m, covariance_matrix=S)
