import math
from typing import Sequence

import torch
from torch import nn

from .bspline import evaluate_local_spline
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
        return evaluate_local_spline(values, grid, coefficients.reshape(
            coefficients.shape[0], *([1] * (values.ndim - 1)), self.coef_size,
        ), self.spline_order)

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

    def __init__(self, input_dim: int, output_dim: int, block_chunk_size: int = 1, **kwargs):
        super().__init__()
        if min(output_dim, block_chunk_size) < 1:
            raise ValueError("output_dim and block_chunk_size must be positive.")
        self.block_chunk_size = block_chunk_size
        self.blocks = nn.ModuleList([
            SprecherBlock(input_dim, **kwargs) for _ in range(output_dim)
        ])
        self.size = sum(block.size for block in self.blocks)

    def forward(self, x, offsets):
        if x.ndim != 2 or x.shape[1] != self.blocks[0].input_dim:
            raise ValueError(f"Expected [N, {self.blocks[0].input_dim}] signals.")
        if offsets.shape != (x.shape[0], self.size):
            raise ValueError(f"Expected [N, {self.size}] curve offsets.")
        pieces = offsets.split([block.size for block in self.blocks], dim=-1)
        outputs = []
        for start in range(0, len(self.blocks), self.block_chunk_size):
            blocks = self.blocks[start:start + self.block_chunk_size]
            if self.block_chunk_size == 1:
                outputs.append(blocks[0](x, pieces[start])[:, None])
                continue
            first = blocks[0]

            psi, xi = torch.stack(pieces[start:start + len(blocks)]).split(first.coef_size, -1)
            psi = psi + torch.stack([block.psi_base for block in blocks])[:, None]
            xi = xi + torch.stack([block.xi_base for block in blocks])[:, None]
            epsilon = torch.stack([block.epsilon for block in blocks])
            q = torch.stack([block.q for block in blocks])
            shifted = x[None, :, None, :] + (epsilon[:, None] * q)[:, None, :, None]
            inner = evaluate_local_spline(
                shifted, torch.stack([block.psi_grid for block in blocks]),
                psi[:, :, None, None], first.spline_order,
            )
            lambdas = torch.stack([block.lambdas.T for block in blocks])
            sums = (inner * lambdas[:, None]).sum(-1)
            outer = evaluate_local_spline(
                sums, torch.stack([block.xi_grid for block in blocks]),
                xi[:, :, None], first.spline_order,
            )
            outputs.append(outer.sum(-1).T)
        return torch.cat(outputs, dim=-1)
