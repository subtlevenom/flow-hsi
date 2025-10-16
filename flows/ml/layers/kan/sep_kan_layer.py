from typing import List
import numpy as np
import torch
from .kan_layer import KANLayer


class SepKANLayer2D(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 residual_std: float = 0.1,
                 grid_range: List[float] = [0, 1]):
        super(SepKANLayer2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kan_layer = KANLayer(in_dim=in_channels,
                                  out_dim=out_channels,
                                  grid_size=grid_size,
                                  spline_order=spline_order,
                                  residual_std=residual_std,
                                  grid_range=grid_range)

        # Arbitrary layers configuration fc
        self._kan_params_indices = [0]

        coef_len = np.prod(self.kan_layer.activation_fn.coef_shape)
        univariate_weight_len = np.prod(
            self.kan_layer.residual_layer.univariate_weight_shape)
        residual_weight_len = np.prod(
            self.kan_layer.residual_layer.residual_weight_shape)
        self._kan_params_indices.extend(
            [coef_len, univariate_weight_len, residual_weight_len])
        self._kan_params_indices = np.cumsum(self._kan_params_indices)

        self.size = self._kan_params_indices[-1]

    def forward(self, x, w):

        B, C, H, W = x.shape

        # weights (b * h * w, kan_size)
        w = w.permute(0, 2, 3, 1).reshape(-1, self.size)
        # img (b * h * w, kan_size)
        x = x.permute(0, 2, 3, 1).reshape(-1, self.in_channels)

        i, j = self._kan_params_indices[0], self._kan_params_indices[1]
        coef = w[:, i:j].view(-1, *self.kan_layer.activation_fn.coef_shape)
        i, j = self._kan_params_indices[1], self._kan_params_indices[2]
        univariate_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.univariate_weight_shape)
        i, j = self._kan_params_indices[2], self._kan_params_indices[3]
        residual_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.residual_weight_shape)
        x = self.kan_layer(x, coef, univariate_weight, residual_weight)
        x = x.view(B, H, W, self.out_channels).permute(0, 3, 1, 2)

        return x


class SepKANLayer1D(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 residual_std: float = 0.1,
                 grid_range: List[float] = [0, 1]):
        super(SepKANLayer1D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kan_layer = KANLayer(in_dim=in_channels,
                                  out_dim=out_channels,
                                  grid_size=grid_size,
                                  spline_order=spline_order,
                                  residual_std=residual_std,
                                  grid_range=grid_range)

        # Arbitrary layers configuration fc
        self._kan_params_indices = [0]

        coef_len = np.prod(self.kan_layer.activation_fn.coef_shape)
        univariate_weight_len = np.prod(
            self.kan_layer.residual_layer.univariate_weight_shape)
        residual_weight_len = np.prod(
            self.kan_layer.residual_layer.residual_weight_shape)
        self._kan_params_indices.extend(
            [coef_len, univariate_weight_len, residual_weight_len])
        self._kan_params_indices = np.cumsum(self._kan_params_indices)

        self.size = self._kan_params_indices[-1]

    def forward(self, x, w):

        B, C, N = x.shape

        # weights (b * h * n, kan_size)
        w = w.permute(0, 2, 1).reshape(-1, self.size)
        # img (b * h * n, kan_size)
        x = x.permute(0, 2, 1).reshape(-1, self.in_channels)

        i, j = self._kan_params_indices[0], self._kan_params_indices[1]
        coef = w[:, i:j].view(-1, *self.kan_layer.activation_fn.coef_shape)
        i, j = self._kan_params_indices[1], self._kan_params_indices[2]
        univariate_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.univariate_weight_shape)
        i, j = self._kan_params_indices[2], self._kan_params_indices[3]
        residual_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.residual_weight_shape)
        x = self.kan_layer(x, coef, univariate_weight, residual_weight)
        x = x.view(B, N, self.out_channels).permute(0, 2, 1)

        return x
