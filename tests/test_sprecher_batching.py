import pytest
import torch

from flows.ml.layers.kan.sprecher import SprecherLayer
from flows.ml.models.cmkan import SprecherNet
from flows.ml.layers.kan.bspline import evaluate_local_spline
from flows.ml.layers.kan.kan_layer import generate_control_points


@pytest.fixture(autouse=True)
def cpu_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(23)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_batched_blocks_match_independent_curves_arguments_and_gradients(batch_size, order):
    layer = SprecherLayer(3, 5, block_chunk_size=1, spline_order=order).double()
    with torch.no_grad():
        for index, block in enumerate(layer.blocks):
            block.psi_base.add_(torch.randn_like(block.psi_base) * 0.05)
            block.xi_base.add_(torch.randn_like(block.xi_base) * 0.05)
            block.lambdas.add_(torch.randn_like(block.lambdas) * 0.02)
            block.epsilon_logit.fill_(index * 0.2 - 0.4)
            # Loaded grids need not be identical across blocks.
            block.psi_grid[..., 2] += index * 0.003
            block.xi_grid[..., 2] += index * 0.002
    x = torch.tensor([[-4., 0.2, 0.9], [0.1, 0.5, 4.]], dtype=torch.double,
                     requires_grad=True)
    offsets = (torch.randn(2, layer.size, dtype=torch.double) * 0.03).requires_grad_()
    inputs = (x, offsets, *layer.parameters())
    expected = layer(x, offsets)
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
    layer.block_chunk_size = batch_size
    actual = layer(x, offsets)
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)


def test_batched_cascade_checkpoint_cache_chunks_and_old_state():
    model = SprecherNet(7, 4, layer_widths=[5, 3], mst_features=8, mst_stages=1,
                        mst_head_channels=8, pixel_chunk_size=7, block_chunk_size=1).double()
    # Strict load of the unchanged scalar layout; batching adds no state.
    state = model.state_dict()
    other = SprecherNet(7, 4, layer_widths=[5, 3], mst_features=8, mst_stages=1,
                        mst_head_channels=8, pixel_chunk_size=3, block_chunk_size=4).double()
    other.load_state_dict(state, strict=True)
    assert state.keys() == other.state_dict().keys()
    x = torch.rand(1, 7, 3, 5, dtype=torch.double)
    params = (torch.randn(1, model.size, 3, 5, dtype=torch.double) * 0.005).requires_grad_()
    expected = model.apply_parameters(x, params)
    expected_grads = torch.autograd.grad(expected.square().sum(),
                                        (params, *model.sprecher_layers.parameters()))
    actual = other.apply_parameters(x, params)
    actual_grads = torch.autograd.grad(actual.square().sum(),
                                      (params, *other.sprecher_layers.parameters()))
    torch.testing.assert_close(actual, expected)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)
    other.eval()
    other.checkpoint_splines = False
    context = torch.rand_like(x)
    torch.testing.assert_close(other.predictor(context)(x),
                               model.apply_parameters(x, model.generate_parameters(context)))
    assert other(x).shape == (1, 4, 3, 5)
    other(x).square().mean().backward()
    assert all(head.weight.grad is not None for head in other.encoder.heads)


def test_batch_size_and_shapes_rejected():
    with pytest.raises(ValueError, match="block_chunk_size"):
        SprecherLayer(3, 4, block_chunk_size=0)
    layer = SprecherLayer(3, 4, block_chunk_size=4)
    with pytest.raises(ValueError, match="signals"):
        layer(torch.rand(2, 4), torch.rand(2, layer.size))
    with pytest.raises(ValueError, match="offsets"):
        layer(torch.rand(2, 3), torch.rand(2, layer.size - 1))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_batched_knots_endpoints_and_support(order):
    grids = torch.stack([generate_control_points(-1 + i * 0.1, 2 + i * 0.1,
                                                 1, 1, order, 7).double()
                         for i in range(3)])
    values = torch.cat((grids.reshape(3, -1),
                        grids.reshape(3, -1) + 1e-7,
                        grids.reshape(3, -1) - 1e-7,
                        torch.tensor([[-5., 5.]]).expand(3, -1)), -1).requires_grad_()
    coefs = torch.randn(3, 1, 7 + order, dtype=torch.double, requires_grad=True)
    expected = torch.stack([evaluate_local_spline(values[i], grids[i], coefs[i], order)
                            for i in range(3)])
    actual = evaluate_local_spline(values, grids, coefs, order)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.square().sum(), (values, coefs), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (values, coefs))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def test_default_batches_all_blocks_without_changing_state():
    scalar = SprecherLayer(3, 5, block_chunk_size=1).double()
    batched = SprecherLayer(3, 5).double()
    batched.load_state_dict(scalar.state_dict(), strict=True)
    assert batched.block_chunk_size == 5
    assert scalar.state_dict().keys() == batched.state_dict().keys()
    assert all(block.epsilon_logit.requires_grad for block in batched.blocks)
    x = torch.rand(7, 3, dtype=torch.double)
    offsets = torch.randn(7, batched.size, dtype=torch.double) * 0.01
    torch.testing.assert_close(batched(x, offsets), scalar(x, offsets))
