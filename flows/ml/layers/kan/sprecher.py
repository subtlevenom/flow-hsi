import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .bspline import evaluate_local_spline
from .shared_spline import SharedSplineLayer

IMPLEMENTATIONS = ("loop", "uniform")


def _uniform_cubic_parts(values, grid, coef_count):
    """Closed-form cardinal cubic B-spline weights on a uniform knot grid.

    Returns padded coefficient positions [..., 4] and their weights [..., 4].
    Positions index coefficients left/right padded by four zeros, which
    reproduces the zero value outside the knot range of evaluate_local_spline.
    """
    start = grid[..., 0]
    step = (grid[..., -1] - start) / (grid.shape[-1] - 1)
    u = ((values - start) / step).clamp(-1, coef_count + 3)
    span = u.detach().floor()
    f = u - span
    g = 1 - f
    f2 = f * f
    f3 = f2 * f
    weights = torch.stack((g * g * g, 3 * f3 - 6 * f2 + 4,
                           -3 * f3 + 3 * f2 + 3 * f + 1, f3), -1) / 6
    positions = span.long()[..., None] + torch.arange(1, 5, device=values.device)
    return positions, weights


def uniform_cubic_spline(values, grid, coefficients):
    """Evaluate per-row cubic splines: values [..., M], coefficients [..., C]."""
    coef_count = coefficients.shape[-1]
    positions, weights = _uniform_cubic_parts(values, grid, coef_count)
    padded = F.pad(coefficients, (4, 4))
    active = padded.gather(-1, positions.flatten(-2)).view_as(weights)
    return (weights * active).sum(-1)


def uniform_cubic_basis(values, grid, coef_count):
    """Dense cubic B-spline basis [..., C] on a uniform knot grid."""
    positions, weights = _uniform_cubic_parts(values, grid, coef_count)
    dense = weights.new_zeros(*values.shape, coef_count + 8)
    return dense.scatter(-1, positions, weights)[..., 4:coef_count + 4]


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
    """m independent scalar Sprecher blocks in parallel, mapping n -> m.

    implementation="loop" evaluates blocks with the generic knot-search
    recurrence (block_chunk_size at a time). implementation="uniform" evaluates
    all blocks at once with closed-form cubic B-splines; it assumes the
    uniform knot grids built at construction and spline_order=3.
    shared_shift=True (uniform only) uses one epsilon for all blocks, so the
    inner basis is computed once per pixel and contracted with all lambdas by
    a single batched matmul (~m times fewer spline evaluations).
    """

    def __init__(self, input_dim: int, output_dim: int, block_chunk_size: int = 1,
                 implementation: str = "loop", shared_shift: bool = False, **kwargs):
        super().__init__()
        if min(output_dim, block_chunk_size) < 1:
            raise ValueError("output_dim and block_chunk_size must be positive.")
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}.")
        if shared_shift and implementation != "uniform":
            raise ValueError("shared_shift requires implementation='uniform'.")
        self.block_chunk_size = block_chunk_size
        self.implementation = implementation
        self.shared_shift = shared_shift
        self.blocks = nn.ModuleList([
            SprecherBlock(input_dim, **kwargs) for _ in range(output_dim)
        ])
        if implementation == "uniform" and self.blocks[0].spline_order != 3:
            raise ValueError("implementation='uniform' requires spline_order=3.")
        if shared_shift:
            self.epsilon_logit = nn.Parameter(torch.tensor(0.0))
            for block in self.blocks:
                block.epsilon_logit.requires_grad_(False)
        self.size = sum(block.size for block in self.blocks)

    def _uniform(self, x, offsets):
        first = self.blocks[0]
        m, coef = len(self.blocks), first.coef_size
        psi, xi = offsets.view(-1, m, 2, coef).unbind(2)
        psi = psi + torch.stack([block.psi_base for block in self.blocks])
        xi = xi + torch.stack([block.xi_base for block in self.blocks])
        lambdas = torch.stack([block.lambdas for block in self.blocks])  # [m, n, Q]
        if self.shared_shift:
            epsilon = (first.max_shift * self.epsilon_logit.sigmoid()
                       / (first.branches - 1))
            shifted = x[:, None, :] + epsilon * first.q[None, :, None]  # [N, Q, n]
            basis = uniform_cubic_basis(shifted, first.psi_grid.reshape(-1), coef)
            # mixed[N, m, Q, C] = sum_p lambda[m, p, Q] * B_C(x_p + eps Q)
            mixed = torch.einsum("nqpc,mpq->nmqc", basis, lambdas)
            sums = (mixed * psi[:, :, None, :]).sum(-1)
        else:
            epsilon = torch.stack([block.epsilon for block in self.blocks])
            shifted = x[:, None, None, :] + (epsilon[:, None] * first.q)[None, :, :, None]
            inner = uniform_cubic_spline(
                shifted.flatten(-2), first.psi_grid.reshape(-1), psi,
            ).view_as(shifted)  # [N, m, Q, n]
            sums = (inner * lambdas.transpose(1, 2)[None]).sum(-1)
        return uniform_cubic_spline(sums, first.xi_grid.reshape(-1), xi).sum(-1)

    def forward(self, x, offsets):
        if x.ndim != 2 or x.shape[1] != self.blocks[0].input_dim:
            raise ValueError(f"Expected [N, {self.blocks[0].input_dim}] signals.")
        if offsets.shape != (x.shape[0], self.size):
            raise ValueError(f"Expected [N, {self.size}] curve offsets.")
        if self.implementation == "uniform":
            return self._uniform(x, offsets)
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
