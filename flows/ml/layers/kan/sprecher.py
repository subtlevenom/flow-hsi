import math
from typing import Sequence

import torch
from torch import nn

from .bspline import compute_bspline
from .shared_spline import SharedSplineLayer


class SprecherBlock(nn.Module):
    """Exact requested sum over q=0..2n and p=1..n; two curves per block."""

    def __init__(
        self, input_dim: int, grid_size: int = 7, spline_order: int = 3,
        psi_range: Sequence[float] = (-1, 2), xi_range: Sequence[float] = (-2, 2),
        max_shift: float = 0.5,
    ):
        super().__init__()
        if input_dim < 1 or not math.isfinite(max_shift) or max_shift <= 0:
            raise ValueError("input_dim and finite max_shift must be positive.")
        self.input_dim = input_dim
        self.branches = 2 * input_dim + 1
        self.spline_order = spline_order
        self.max_shift = max_shift
        # Reuse cmKAN knot construction and Greville initialization, not its neuron.
        psi = SharedSplineLayer(1, 1, grid_size, spline_order, psi_range)
        xi = SharedSplineLayer(1, 1, grid_size, spline_order, xi_range)
        self.coef_size = psi.coef_size
        self.size = 2 * self.coef_size
        self.register_buffer("psi_grid", psi.grid)
        self.register_buffer("xi_grid", xi.grid)
        self.psi_base = nn.Parameter(psi.base_parameters[:self.coef_size].detach().clone())
        self.xi_base = nn.Parameter(
            xi.base_parameters[:self.coef_size].detach().clone() / self.branches,
        )
        self.lambdas = nn.Parameter(torch.full((input_dim, self.branches), 1 / input_dim))
        self.epsilon_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("q", torch.arange(self.branches, dtype=torch.float32))

    @property
    def epsilon(self):
        # One scalar for all q, with the largest shift bounded by max_shift.
        return self.max_shift * self.epsilon_logit.sigmoid() / (self.branches - 1)

    def curve(self, values, coefficients, grid):
        bases = compute_bspline(
            values.reshape(-1, 1), grid.to(dtype=values.dtype), self.spline_order,
        ).reshape(*values.shape, self.coef_size)
        return (bases * coefficients.reshape(
            coefficients.shape[0], *([1] * (values.ndim - 1)), self.coef_size,
        )).sum(dim=-1)

    def forward(self, x, offsets):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected [N, {self.input_dim}] signals.")
        if offsets.shape != (x.shape[0], self.size):
            raise ValueError(f"Expected [N, {self.size}] curve offsets.")
        psi, xi = offsets.split(self.coef_size, dim=-1)
        shifted = x[:, None, :] + self.epsilon * self.q[None, :, None]
        inner = self.curve(shifted, self.psi_base + psi, self.psi_grid)
        sums = (inner * self.lambdas.T[None]).sum(dim=-1)
        return self.curve(sums, self.xi_base + xi, self.xi_grid).sum(dim=-1)


class SprecherLayer(nn.Module):
    """m independent scalar Sprecher blocks in parallel, mapping n -> m."""

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive.")
        self.blocks = nn.ModuleList([
            SprecherBlock(input_dim, **kwargs) for _ in range(output_dim)
        ])
        self.size = sum(block.size for block in self.blocks)

    def forward(self, x, offsets):
        pieces = offsets.split([block.size for block in self.blocks], dim=-1)
        return torch.stack([block(x, part) for block, part in zip(self.blocks, pieces)], dim=-1)
