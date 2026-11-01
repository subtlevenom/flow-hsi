import torch


def evaluate_local_spline(values, grid, coefficients, order):
    """Evaluate a shared knot grid using only order+1 active coefficients.

    Coefficients broadcast over values' leading dimensions. A batch of knot
    grids uses values' first dimension for independent curves. Invalid edge
    coefficients are zero, preserving the original finite-support recurrence.
    """
    batched = grid.numel() != grid.shape[-1]
    knots = grid.to(device=values.device, dtype=values.dtype)
    if batched:
        knots = knots.reshape(values.shape[0], -1)
        span = torch.searchsorted(
            knots, values.reshape(values.shape[0], -1).contiguous(), right=True,
        ).reshape_as(values) - 1
    else:
        knots = knots.reshape(-1)
        span = torch.searchsorted(knots, values.contiguous(), right=True) - 1
    knot_count = knots.shape[-1]
    indices = span[..., None] - order + torch.arange(order + 1, device=values.device)

    def knot(index):
        bounded = index.clamp(0, knot_count - 1)
        if batched:
            shape = (values.shape[0], *([1] * (index.ndim - 1)))
            table = knots.reshape(*shape[:-1], knot_count).expand(*index.shape[:-1], knot_count)
            return (table.gather(-1, bounded)
                    + index.clamp(max=0) * (knots[:, 1] - knots[:, 0]).reshape(shape)
                    + (index - knot_count + 1).clamp(min=0)
                    * (knots[:, -1] - knots[:, -2]).reshape(shape))
        # Virtual knots only serve bases with nonexistent, zero coefficients.
        return (knots[bounded]
                + index.clamp(max=0) * (knots[1] - knots[0])
                + (index - knot_count + 1).clamp(min=0) * (knots[-1] - knots[-2]))

    if order == 3:
        # Unroll cubic recurrence; use actual knots, including their rounding,
        # rather than assuming an exactly uniform cardinal spline.
        t0, t1, t2, t3, t4, t5 = knot(
            span[..., None] + torch.arange(-2, 4, device=values.device),
        ).unbind(-1)
        a = (t3 - values) / (t3 - t2)
        b = (values - t2) / (t3 - t2)
        c = (t3 - values) / (t3 - t1) * a
        d = (values - t1) / (t3 - t1) * a + (t4 - values) / (t4 - t2) * b
        e = (values - t2) / (t4 - t2) * b
        bases = torch.stack((
            (t3 - values) / (t3 - t0) * c,
            (values - t0) / (t3 - t0) * c + (t4 - values) / (t4 - t1) * d,
            (values - t1) / (t4 - t1) * d + (t5 - values) / (t5 - t2) * e,
            (values - t2) / (t5 - t2) * e,
        ), -1)
    else:
        bases = torch.ones_like(values)[..., None]
        for degree in range(1, order + 1):
            current = indices[..., order - degree:]
            left = (values[..., None] - knot(current)) / (knot(current + degree) - knot(current))
            right = (knot(current + degree + 1) - values[..., None]) / (
                knot(current + degree + 1) - knot(current + 1))
            zeros = torch.zeros_like(bases[..., :1])
            bases = left * torch.cat((zeros, bases), -1) + right * torch.cat((bases, zeros), -1)
    valid = (indices >= 0) & (indices < coefficients.shape[-1])
    expanded = coefficients.expand(*values.shape, coefficients.shape[-1])
    active = expanded.gather(-1, indices.clamp(0, coefficients.shape[-1] - 1))
    return (bases * active * valid).sum(-1)

# Helper functions for computing B splines over a grid
def compute_bspline(x: torch.Tensor, grid: torch.Tensor, k: int):
    """
    For a given grid with G_1 intervals and spline order k, we *recursively* compute
    and evaluate each B_n(x_{ij}). x is a (batch_size, in_dim) and grid is a
    (out_dim, in_dim, # grid points + 2k + 1)

    Returns a (batch_size, out_dim, in_dim, grid_size + k) intermediate tensor to 
    compute sum_i {c_i B_i(x)} with.

    """
    
    grid = grid[None, :, :, :].to(x.device)
    x = x[:, None, :, None].to(x.device)
    
    # Base case: B_{i,0}(x) = 1 if (grid_i <= x <= grid_{i+k}) 0 otherwise
    bases = (x >= grid[:, :, :, :-1]) * (x < grid[:, :, :, 1:])

    # Recurse over spline order j, vectorize over basis function i
    for j in range (1, k + 1):
        n = grid.size(-1) - (j + 1)
        b1 = ((x[:, :, :, :] - grid[:, :, :, :n]) / (grid[:, :, :, j:-1] - grid[:, :, :, :n])) * bases[:, :, :, :-1]
        b2 = ((grid[:, :, :, j+1:] - x[:, :, :, :])  / (grid[:, :, :, j+1:] - grid[:, :, :, 1:n+1])) * bases[:, :, :, 1:]
        bases = b1 + b2

    return bases

def coef2curve (x : torch.Tensor, grid: torch.Tensor, coefs: torch.Tensor, k: int, device:torch.device):
    """
    For a given (batch of) x, control points (grid), and B-spline coefficients,
    evaluate and return x on the B-spline function.
    """
    bases = compute_bspline(x, grid, k, device)
    spline = torch.sum(bases * coefs[None, ...], dim=-1)
    return spline


if __name__ == "__main__":
    print("B Spline Unit Tests")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bsz = 2
    spline_order = 3
    in_dim = 5
    out_dim = 7
    grid_size = 11
    grid_range = [-1.,1]

    x = torch.ones(bsz, in_dim) / 0.8

    spacing = (grid_range[1] - grid_range[0]) / grid_size
    grid = torch.arange(-spline_order, grid_size + spline_order + 1, device=device) * spacing + grid_range[0]
    # Generate (out, in) copies of equally spaced points on [a, b]
    grid = grid[None, None, ...].expand(out_dim, in_dim, -1).contiguous()

    print('x', x)
    print('grid', grid)

    compute_bspline(x, grid, spline_order, device)
