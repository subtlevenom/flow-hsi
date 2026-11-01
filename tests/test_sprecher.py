from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flows.cli.train import train_default
from flows.core import ModelSelector
from flows.ml.layers.kan.bspline import compute_bspline
from flows.ml.layers.kan.sprecher import SprecherBlock, SprecherLayer
from flows.ml.models.cmkan import SprecherNet
from flows.ml.layers.encoders.cm_encoder import FFN
from flows.ml.datasets.ntire import datamodule as ntire


@pytest.fixture(autouse=True)
def cpu_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(5)
    yield
    torch.set_num_threads(threads)


def configuration(overrides=()):
    with initialize_config_dir(version_base="1.3.2",
                               config_dir=str(Path(__file__).resolve().parents[1] / ".configs")):
        return compose(config_name="sprecher", overrides=list(overrides))


def test_exact_equation_and_gradients():
    block = SprecherBlock(2).double()
    with torch.no_grad():
        block.lambdas.add_(torch.randn_like(block.lambdas) * 0.04)
    x = torch.tensor([[0.2, 0.6], [0.4, 0.7]], dtype=torch.double, requires_grad=True)
    offsets = (torch.randn(2, block.size, dtype=torch.double) * 0.02).requires_grad_()

    def curve(t, grid, coef):
        return (compute_bspline(t.reshape(1, 1), grid, 3).flatten() * coef).sum()

    expected = []
    for row in range(2):
        total = x.new_zeros(())
        for q in range(5):
            inner = x.new_zeros(())
            for p in range(2):
                inner = inner + block.lambdas[p, q] * curve(
                    x[row, p] + block.epsilon * q, block.psi_grid,
                    block.psi_base + offsets[row, :10],
                )
            total = total + curve(inner, block.xi_grid, block.xi_base + offsets[row, 10:])
        expected.append(total)
    torch.testing.assert_close(block(x, offsets), torch.stack(expected))
    assert block.lambdas.shape == (2, 5)
    assert 0 < block.epsilon < block.max_shift / 4
    assert torch.autograd.gradcheck(block, (x, offsets))
    block(x, offsets).square().mean().backward()
    assert block.lambdas.grad.abs().sum() > 0
    assert block.epsilon_logit.grad.abs() > 0
    assert all(part.abs().sum() > 0 for part in offsets.grad.split(10, -1))


def test_parallel_independence():
    layer = SprecherLayer(2, 3)
    x = torch.rand(4, 2)
    offsets = torch.zeros(4, 60)
    original = layer(x, offsets)
    offsets[:, 20:30] = 0.1
    changed = layer(x, offsets)
    torch.testing.assert_close(original[:, [0, 2]], changed[:, [0, 2]])
    assert not torch.allclose(original[:, 1], changed[:, 1])
    assert len({b.lambdas.data_ptr() for b in layer.blocks}) == 3
    assert len({b.epsilon_logit.data_ptr() for b in layer.blocks}) == 3


@pytest.mark.parametrize("channels,output,hw", [(7, 4, (9, 11)), (31, 31, (1, 1))])
def test_cascade_generator_gradients_cache_and_chunk(channels, output, hw):
    model = SprecherNet(channels, output, layer_widths=[3, 2], mst_features=8,
                        mst_stages=1, mst_head_channels=8, pixel_chunk_size=7)
    x = torch.rand(1, channels, *hw)
    assert len(model.encoder.heads) == 5
    assert all(head.out_channels == 20 for head in model.encoder.heads)
    assert isinstance(model.output_ffn, FFN)
    assert model.generate_parameters(x).shape == (1, 100, *hw)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        y = model(x)
        assert y.shape == (1, output, *hw)
        (y - 0.3).square().mean().backward()
        optimizer.step()
    for head in model.encoder.heads:
        assert head.weight.grad[:10].abs().sum() > 0
        assert head.weight.grad[10:].abs().sum() > 0
    assert model.encoder.stem.weight.grad.abs().sum() > 0
    for layer in model.sprecher_layers:
        for block in layer.blocks:
            assert block.lambdas.grad.abs().sum() > 0
            assert block.epsilon_logit.grad.abs() > 0
    model.eval()
    context = torch.rand_like(x)
    cached = model.predictor(context)
    torch.testing.assert_close(cached(x), model.apply_parameters(x, model.generate_parameters(context)))
    expected = model(x)
    model.pixel_chunk_size = 10000
    torch.testing.assert_close(model(x), expected)
    with pytest.raises(ValueError, match="spatial grid"):
        cached(torch.rand(1, channels, hw[0] + 1, hw[1]))


def test_actual_cli_default_paths(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    cube = np.full((128, 128, 31), 0.5, dtype=np.float32)
    for folder in (source, target):
        np.save(folder / "sample.npy", cube)
    paths = []
    original = ntire.paired_paths

    def paired(a, b):
        paths.append((a, b))
        return original(str(source), str(target))

    def fit(model, datamodule, ckpt_path):
        datamodule.setup("fit")
        assert isinstance(model.model.layers.cmkan, SprecherNet)
        assert datamodule.train_dataset[0][0].shape == (31, 32, 32)

    monkeypatch.setattr(ntire, "paired_paths", paired)
    monkeypatch.setattr("flows.cli.train.L.Trainer", lambda **kwargs: SimpleNamespace(fit=fit))
    monkeypatch.setattr(sys, "argv", ["main.py", "--config-name", "sprecher", "task=train"])
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")
    root = configuration().dataset_root
    assert paths[0] == (f"{root}/train/source", f"{root}/train/target")


def test_config_and_bounded_training(tmp_path):
    config = configuration()
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    model = ModelSelector.select(config.model).layers.cmkan
    assert model.input_dim == model.output_dim == 31
    assert model.parameter_sizes == (620,) and model.size == 620
    assert [len(layer.blocks) for layer in model.sprecher_layers] == [31]
    assert [layer.blocks[0].branches for layer in model.sprecher_layers] == [63]
    assert len(model.encoder.heads) == 31
    assert all(head.out_channels == 20 for head in model.encoder.heads)
    blocks = model.sprecher_layers[0].blocks
    assert all(block.lambdas.shape == (31, 63) for block in blocks)
    assert sum(block.lambdas.numel() for block in blocks) == 60543
    assert sum(block.epsilon_logit.numel() for block in blocks) == 31
    assert model.output_ffn.pointwise1.in_channels == 31
    assert model.output_ffn.pointwise1.out_channels == 32
    assert model.output_ffn.pointwise2.out_channels == 31
    x = torch.rand(1, 31, 3, 5)
    curves = model.generate_parameters(x)
    assert curves.shape == (1, 620, 3, 5)
    model.apply_parameters(x, curves).square().mean().backward()
    for block, head in zip(blocks, model.encoder.heads):
        assert block.lambdas.grad.abs().sum() > 0
        assert block.epsilon_logit.grad.abs() > 0
        assert head.weight.grad[:10].abs().sum() > 0
        assert head.weight.grad[10:].abs().sum() > 0
    for split in ("train", "test"):
        for domain in ("source", "target"):
            folder = tmp_path / "dataset" / split / domain
            folder.mkdir(parents=True)
            np.save(folder / "sample.npy", np.random.default_rng(3).uniform(
                0.1, 0.9, (8, 8, 31),
            ).astype("float32"))
    config = configuration([
        f"dataset_root={(tmp_path / 'dataset').as_posix()}",
        f"save_dir={(tmp_path / 'experiments').as_posix()}",
        "accelerator=cpu", "devices=1", "epochs=1",
        "data.params.num_workers=0", "data.params.pin_memory=false",
        "data.params.train.crop_size=8", "data.params.val.crop_size=8",
        "model.params.layers.cmkan.params.mst_features=8",
        "model.params.layers.cmkan.params.mst_stages=1",
        "model.params.layers.cmkan.params.mst_head_channels=8",
    ])
    train_default(config)
    path = tmp_path / "experiments" / config.experiment / "logs" / "checkpoints" / "last.ckpt"
    state = torch.load(path, weights_only=False)
    assert state["global_step"] == 1
    assert all(torch.isfinite(v).all() for v in state["state_dict"].values())
    overridden = configuration(["spectral_channels=7",
                                "model.params.layers.cmkan.params.grid_size=5",
                                "model.params.layers.cmkan.params.spline_order=2",
                                "model.params.layers.cmkan.params.implementation=loop"])
    other = ModelSelector.select(overridden.model).layers.cmkan
    assert other.input_dim == other.output_dim == 7
    assert other.size == 31 * 2 * 7
