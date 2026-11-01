import pytest
import torch

from flows.ml.layers.kan.bspline import compute_bspline, evaluate_local_spline
from flows.ml.layers.kan.kan_layer import generate_control_points
from flows.ml.layers.kan.shared_spline import SharedSplineLayer
from flows.ml.models.cmkan import SharedCmKAN


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_matches_dense_at_knots_and_finite_support(order, dtype):
    grid = generate_control_points(-1, 2, 1, 1, order, 7).to(dtype)
    # Also cover nonuniform loaded grids; no uniform-knot approximation.
    grid[..., 2] += 0.02
    knots = grid.flatten()
    values = torch.cat((knots, knots - 1e-5, knots + 1e-5,
                        torch.linspace(-5, 6, 73, dtype=dtype))).reshape(1, -1)
    values.requires_grad_()
    coefficients = torch.randn(1, 1, 7 + order, dtype=dtype, requires_grad=True)
    local = evaluate_local_spline(values, grid, coefficients, order)
    dense = (compute_bspline(values.reshape(-1, 1), grid, order)
             .reshape(1, -1, 7 + order) * coefficients).sum(-1)
    torch.testing.assert_close(local, dense)
    local_grads = torch.autograd.grad(local.square().sum(), (values, coefficients), retain_graph=True)
    dense_grads = torch.autograd.grad(dense.square().sum(), (values, coefficients))
    for actual, expected in zip(local_grads, dense_grads):
        torch.testing.assert_close(actual, expected)
    assert local[0, -1] == 0 and local[0, -73] == 0


def test_local_broadcast_and_gradcheck():
    grid = generate_control_points(0, 1, 1, 1, 3, 7).double()
    values = torch.rand(2, 3, 4, dtype=torch.double, requires_grad=True)
    coefficients = torch.randn(2, 1, 1, 10, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, c: evaluate_local_spline(x, grid, c, 3),
                                   (values, coefficients))


@pytest.mark.parametrize("shape", [(10,), (1, 1, 1, 10), (2, 1, 1, 10),
                                    (1, 3, 1, 10), (2, 3, 4, 10)])
def test_compact_gathers_preserve_coefficient_broadcasting(shape):
    grid = generate_control_points(-1, 2, 1, 1, 3, 7).double()
    values = torch.rand(2, 3, 4, dtype=torch.double, requires_grad=True)
    # A noncontiguous view must also propagate into its original storage.
    storage = torch.randn(*shape[:-1], 20, dtype=torch.double, requires_grad=True)
    coefficients = storage[..., ::2]
    local = evaluate_local_spline(values, grid, coefficients, 3)
    dense = (compute_bspline(values.reshape(-1, 1), grid, 3)
             .reshape(*values.shape, 10) * coefficients).sum(-1)
    torch.testing.assert_close(local, dense)
    a = torch.autograd.grad(local.square().sum(), (values, storage), retain_graph=True)
    b = torch.autograd.grad(dense.square().sum(), (values, storage))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)


def test_shared_layer_generated_argument_and_base_gradients():
    layer = SharedSplineLayer(3, 4, grid_size=7).double()
    x = torch.rand(5, 3, dtype=torch.double, requires_grad=True)
    offsets = (torch.randn(5, layer.size, dtype=torch.double) * 0.1).requires_grad_()
    coefs, weights, shifts, biases = layer.unpack(layer.base_parameters + offsets)
    shifted = x[:, None, :] + shifts[:, :, None]
    activated = (compute_bspline(shifted.reshape(-1, 1), layer.grid.double(), 3)
                 .reshape(5, 4, 3, 10) * coefs[:, None, None]).sum(-1)
    expected = biases + (weights * activated).sum(-1)
    actual = layer(x, offsets)
    torch.testing.assert_close(actual, expected)
    inputs = (x, offsets, layer.base_parameters)
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)


def test_checkpoint_chunk_gradients_and_state_roundtrip():
    model = SharedCmKAN(7, 7, hidden_channels=[8, 8], encoder_type="mst",
                        mst_features=8, mst_stages=1, mst_head_channels=8, grid_size=7,
                        pixel_chunk_size=5).double()
    x = torch.rand(1, 7, 3, 5, dtype=torch.double)
    parameters = (torch.randn(1, model.size, 3, 5, dtype=torch.double) * 0.01).requires_grad_()
    result = model.apply_parameters(x, parameters)
    expected_grads = torch.autograd.grad(result.square().sum(),
                                        (parameters, *model.spline_layers.parameters()))
    model.pixel_chunk_size = 1000
    model.checkpoint_splines = False
    actual = model.apply_parameters(x, parameters)
    torch.testing.assert_close(actual, result)
    actual_grads = torch.autograd.grad(actual.square().sum(),
                                      (parameters, *model.spline_layers.parameters()))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)
    model.load_state_dict(model.state_dict(), strict=True)
