from typing import Sequence

import torch
from torch import nn

from .bspline import compute_bspline
from .kan_layer import generate_control_points


class SharedSplineLayer(nn.Module):
    """y_j = c_j + sum_i W_ji psi(x_i + b_j), with one curve per layer.

    A conditioning generator may supply per-sample parameter offsets, but the
    curve is always shared across all input edges and receiving neurons.
    """

    def __init__(
        self, in_dim: int, out_dim: int, grid_size: int = 5,
        spline_order: int = 3, grid_range: Sequence[float] = (0, 1),
    ):
        super().__init__()
        if min(in_dim, out_dim, grid_size, spline_order) < 1:
            raise ValueError("Dimensions, grid_size and spline_order must be positive.")
        bounds = torch.as_tensor(grid_range)
        if bounds.shape != (2,) or not torch.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("grid_range must contain two finite increasing bounds.")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.spline_order = spline_order
        self.coef_size = grid_size + spline_order
        self.parameter_sizes = (self.coef_size, out_dim * in_dim, out_dim, out_dim)
        self.size = sum(self.parameter_sizes)
        grid = generate_control_points(*grid_range, 1, 1, spline_order, grid_size)
        self.register_buffer("grid", grid)
        # Greville abscissae reproduce psi(t)=t inside the original grid range.
        coefficients = torch.stack([
            grid[0, 0, i + 1:i + spline_order + 1].mean()
            for i in range(self.coef_size)
        ])
        weights = torch.eye(out_dim, in_dim)
        self.base_parameters = nn.Parameter(torch.cat([
            coefficients, weights.flatten(), torch.zeros(2 * out_dim),
        ]))

    def unpack(self, parameters: torch.Tensor):
        if parameters.shape[-1] != self.size:
            raise ValueError(f"Expected {self.size} spline/W/shift/bias parameters.")
        coefficients, weights, shifts, biases = parameters.split(self.parameter_sizes, dim=-1)
        return coefficients, weights.reshape(-1, self.out_dim, self.in_dim), shifts, biases

    def forward(self, x: torch.Tensor, offsets: torch.Tensor = None):
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected [N, {self.in_dim}] signals.")
        if offsets is not None and offsets.shape != (x.shape[0], self.size):
            raise ValueError(f"Expected [N, {self.size}] generated offsets.")
        parameters = self.base_parameters.unsqueeze(0).expand(x.shape[0], -1)
        if offsets is not None:
            parameters = parameters + offsets
        coefficients, weights, shifts, biases = self.unpack(parameters)
        shifted = x[:, None, :] + shifts[:, :, None]
        # Flatten edges solely to reuse cmKAN's existing Cox-de Boor recurrence.
        bases = compute_bspline(
            shifted.reshape(-1, 1), self.grid.to(dtype=x.dtype), self.spline_order,
        ).reshape(x.shape[0], self.out_dim, self.in_dim, self.coef_size)
        activated = (bases * coefficients[:, None, None, :]).sum(dim=-1)
        return biases + (weights * activated).sum(dim=-1)
