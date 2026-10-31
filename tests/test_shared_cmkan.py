from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import MISSING, OmegaConf

from flows.cli.train import train_default
from flows.core import DataSelector, ModelSelector, PipelineSelector
from flows.ml.layers.kan.bspline import compute_bspline
from flows.ml.layers.kan.shared_spline import SharedSplineLayer
from flows.ml.layers.mst import MSTParameterGenerator
from flows.ml.models.cmkan import SharedCmKAN
from flows.ml.pipelines.shared_cmkan import SharedCmKANPipeline
from flows.ml.datasets.ntire import datamodule as ntire_datamodule


@pytest.fixture(autouse=True)
def cpu_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(threads)


def configuration(**values):
    with initialize_config_dir(
        version_base="1.3.2", config_dir=str(Path(__file__).resolve().parents[1] / ".configs"),
    ):
        return compose(config_name="shared_cmkan",
                       overrides=[f"{k}={v}" for k, v in values.items()])


def test_default_ntire_paths_and_optional_root_override():
    config = configuration()
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    existing = OmegaConf.load(
        Path(__file__).resolve().parents[1] / ".configs" / "flows" / "data" / "ntire.yaml",
    )
    assert resolved["data"]["name"] == existing.name
    for split in ("train", "val"):
        for domain in ("source", "target"):
            assert resolved["data"]["params"][split][domain] == existing.params[split][domain]
    for setting in ("num_workers", "persistent_workers", "prefetch_factor", "pin_memory", "seed"):
        assert resolved["data"]["params"][setting] == existing.params[setting]
    overridden = configuration(dataset_root="/custom/NTIRE/dataset")
    for split, folder in (("train", "train"), ("val", "test"), ("test", "test")):
        for domain in ("source", "target"):
            assert overridden.data.params[split][domain] == f"/custom/NTIRE/dataset/{folder}/{domain}"


@pytest.mark.parametrize("override", [None, "/custom/NTIRE/dataset"])
def test_actual_cli_reaches_datamodule_setup_without_dropping_root(tmp_path, monkeypatch, override):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    cube = np.full((128, 128, 31), 0.5, dtype=np.float32)
    np.save(source / "sample.npy", cube)
    np.save(target / "sample.npy", cube)
    requested_paths = []
    original_paired_paths = ntire_datamodule.paired_paths

    def paired_paths(source_path, target_path):
        requested_paths.append((source_path, target_path))
        return original_paired_paths(str(source), str(target))

    def fit(model, datamodule, ckpt_path):
        datamodule.setup("fit")
        assert datamodule.train_dataset[0][0].shape == (31, 32, 32)
        assert datamodule.val_dataset[0][0].shape == (31, 128, 128)

    monkeypatch.setattr(ntire_datamodule, "paired_paths", paired_paths)
    monkeypatch.setattr("flows.cli.train.L.Trainer",
                        lambda **kwargs: SimpleNamespace(fit=fit))
    argv = ["main.py", "--config-name", "shared_cmkan", "task=train"]
    if override is not None:
        argv.append(f"dataset_root={override}")
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")
    expected_root = override or configuration().dataset_root
    assert requested_paths == [
        (f"{expected_root}/train/source", f"{expected_root}/train/target"),
        (f"{expected_root}/test/source", f"{expected_root}/test/target"),
    ]


def test_missing_root_reproduces_reported_setup_failure():
    from omegaconf.errors import InterpolationToMissingValueError

    config = configuration()
    config.dataset_root = MISSING
    dm = DataSelector.select(config.data)
    with pytest.raises(InterpolationToMissingValueError, match="dataset_root"):
        dm.setup("fit")


def test_neuron_matches_explicit_shift_before_spline_weighted_sum():
    layer = SharedSplineLayer(3, 2).double()
    x = torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], dtype=torch.float64,
                     requires_grad=True)
    offsets = (torch.randn(2, layer.size, dtype=torch.float64) * 0.03).requires_grad_()
    coefficients, weights, shifts, biases = layer.unpack(layer.base_parameters + offsets)
    expected = []
    for n in range(2):
        receivers = []
        for j in range(2):
            result = biases[n, j]
            for i in range(3):
                basis = compute_bspline((x[n, i] + shifts[n, j]).reshape(1, 1),
                                        layer.grid, layer.spline_order).flatten()
                result = result + weights[n, j, i] * (basis * coefficients[n]).sum()
            receivers.append(result)
        expected.append(torch.stack(receivers))
    expected = torch.stack(expected)
    actual = layer(x, offsets)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(layer, (x, offsets))
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(offsets.grad).all()
    assert all(piece.abs().sum() > 0
               for piece in offsets.grad.split(layer.parameter_sizes, dim=-1))
    assert layer.base_parameters.numel() == 8 + 2 * 3 + 2 * 2
    assert layer.grid.shape[:2] == (1, 1)


def test_identity_initialization_and_shared_coefficient_gradient():
    layer = SharedSplineLayer(4, 4)
    x = torch.linspace(0, 1, 24).reshape(6, 4)
    torch.testing.assert_close(layer(x), x)
    assert "grid" in layer.state_dict()
    layer(x).sum().backward()
    assert layer.base_parameters.grad[:layer.coef_size].abs().sum() > 0


@pytest.mark.parametrize("channels,out_channels", [(3, 5), (7, 7), (31, 31)])
@pytest.mark.parametrize("encoder_type", ["light", "full"])
def test_generator_shapes_composition_and_backward(channels, out_channels, encoder_type):
    model = SharedCmKAN(channels, out_channels, hidden_channels=[channels],
                        conditioning_channels=4, encoder_type=encoder_type,
                        pixel_chunk_size=19)
    x = torch.rand(1, channels, 9, 11, requires_grad=True)
    parameters = model.generate_parameters(x)
    expected_size = sum(10 + i * o + 2 * o
                        for i, o in [(channels, channels), (channels, out_channels)])
    assert parameters.shape == (1, expected_size, 9, 11)
    assert not parameters.count_nonzero()
    result = model.apply_parameters(x, parameters)
    assert result.shape == (1, out_channels, 9, 11)
    if channels == out_channels:
        torch.testing.assert_close(result, x)
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()
    (model(x) - 0.5).square().mean().backward()
    assert model.conditioning_projection.weight.grad.abs().sum() > 0
    assert model.encoder.basis.grad.abs().sum() > 0


@pytest.mark.parametrize("hidden,sizes", [
    (None, (1594, 2410, 1560)),
    ("[48,64,48,32]", (1594, 3210, 3178, 1610, 1064)),
])
@pytest.mark.parametrize("encoder_type", [None, "light", "full"])
def test_configured_depth_generator_shapes_and_backward(hidden, sizes, encoder_type):
    options = {}
    if encoder_type is not None:
        options["model.params.layers.cmkan.params.encoder_type"] = encoder_type
    if hidden is not None:
        options["model.params.layers.cmkan.params.hidden_channels"] = hidden
    config = configuration(**options)
    model = ModelSelector.select(config.model).layers.cmkan
    assert model.encoder_type == (encoder_type or "mst")
    assert isinstance(model.encoder, MSTParameterGenerator) == (encoder_type is None)
    widths = [31, *config.model.params.layers.cmkan.params.hidden_channels, 31]
    assert widths == ([31, 48, 48, 31] if hidden is None else [31, 48, 64, 48, 32, 31])
    assert len(model.spline_layers) == len(sizes)
    assert model.parameter_sizes == sizes and model.size == sum(sizes)
    x = torch.rand(1, 31, 6, 7, requires_grad=True)
    parameters = model.generate_parameters(x)
    assert parameters.shape == (1, sum(sizes), 6, 7)
    result = model.apply_parameters(x, parameters)
    torch.testing.assert_close(result, x)
    (result - 0.5).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_cached_conditioning_chunk_equivalence_and_checkpoint_roundtrip(tmp_path):
    model = SharedCmKAN(5, 5, hidden_channels=[5], conditioning_channels=4,
                        pixel_chunk_size=7)
    with torch.no_grad():
        model.encoder.conv_reproj.pointwise2.weight.normal_(std=0.01)
    context, x = torch.rand(1, 5, 8, 12), torch.rand(1, 5, 8, 12)
    model.eval()
    parameters = model.generate_parameters(context)
    prediction = model.predictor(context)
    torch.testing.assert_close(prediction(x), model.apply_parameters(x, parameters))
    assert not torch.allclose(parameters, model.generate_parameters(x))
    expected = model(x)
    model.pixel_chunk_size = 10000
    torch.testing.assert_close(model(x), expected)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    restored = SharedCmKAN(5, 5, hidden_channels=[5], conditioning_channels=4)
    restored.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(x), expected)
    with pytest.raises(ValueError, match="spatial grid"):
        prediction(x[..., :4, :4])


@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (1, 3, 2, 5)])
@pytest.mark.parametrize("encoder_type", ["light", "mst"])
def test_tiny_signals(shape, encoder_type):
    model = SharedCmKAN(3, 3, conditioning_channels=4, encoder_type=encoder_type,
                        mst_features=8, mst_stages=1, mst_head_channels=8)
    x = torch.rand(shape)
    torch.testing.assert_close(model(x), x)


def small_mst(channels, out_channels, **kwargs):
    return SharedCmKAN(channels, out_channels, hidden_channels=[6], encoder_type="mst",
                       mst_features=8, mst_stages=2, mst_head_channels=16, **kwargs)


def test_mst_feature_and_parameter_shapes_with_default_widths():
    model = SharedCmKAN(31, 31, hidden_channels=[48, 48], encoder_type="mst")
    generator = model.encoder
    assert len(generator.heads) == 3
    assert [h.out_channels for h in generator.heads] == [1594, 2410, 1560]
    assert not hasattr(model, "conditioning_projection")
    x = torch.rand(1, 31, 9, 11)
    features = generator.features(x)
    assert features.shape == (1, 32, 9, 11)
    parameters = model.generate_parameters(x)
    assert parameters.shape == (1, 5564, 9, 11)
    assert not parameters.count_nonzero()
    torch.testing.assert_close(model.apply_parameters(x, parameters), x)


@pytest.mark.parametrize("channels,out_channels", [(7, 7), (7, 4), (31, 31)])
@pytest.mark.parametrize("hw", [(1, 1), (9, 11), (7, 13)])
def test_mst_gradients_reach_backbone_heads_and_splines(channels, out_channels, hw):
    model = small_mst(channels, out_channels, pixel_chunk_size=17)
    x = torch.rand(1, channels, *hw)
    parameters = model.generate_parameters(x)
    assert parameters.shape == (1, model.size, *hw)
    assert model.size == sum(10 + i * o + 2 * o
                             for i, o in [(channels, 6), (6, out_channels)])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(2):
        optimizer.zero_grad()
        y = model(x)
        assert y.shape == (1, out_channels, *hw)
        (y - 0.3).square().mean().backward()
        optimizer.step()
    generator = model.encoder
    for module in (generator.stem, generator.shared_head[0], *generator.heads):
        assert module.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in generator.body.parameters())
    for layer in model.spline_layers:
        assert layer.base_parameters.grad[:layer.coef_size].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_mst_one_untied_generated_curve_per_layer():
    config = configuration()
    params = config.model.params.layers.cmkan.params
    assert (params.grid_size, params.spline_order) == (7, 3)
    model = ModelSelector.select(config.model).layers.cmkan
    assert [layer.coef_size for layer in model.spline_layers] == [10, 10, 10]
    assert model.parameter_sizes == (1594, 2410, 1560) and model.size == 5564
    x = torch.rand(1, 31, 5, 6)
    parameters = model.generate_parameters(x).requires_grad_()
    parameters.retain_grad()
    (model.apply_parameters(x, parameters) - 0.3).square().mean().backward()
    slices = parameters.grad.split(model.parameter_sizes, dim=1)
    for layer, head, grad in zip(model.spline_layers, model.encoder.heads, slices):
        assert grad[:, :layer.coef_size].abs().sum() > 0
        assert head.weight.grad[:layer.coef_size].abs().sum() > 0
    # Each head only drives its own layer's slice, so curves are never tied.
    with torch.no_grad():
        model.encoder.heads[1].bias[:10] = 0.05
    shifted = model.generate_parameters(x).split(model.parameter_sizes, dim=1)
    assert not shifted[0].count_nonzero() and not shifted[2].count_nonzero()
    assert torch.allclose(shifted[1][:, :10], torch.full_like(shifted[1][:, :10], 0.05))
    assert not shifted[1][:, 10:].count_nonzero()
    coefficients = [layer.base_parameters[:layer.coef_size] for layer in model.spline_layers]
    assert len({c.data_ptr() for c in coefficients}) == 3


def test_mst_cached_predictor_and_chunk_equivalence():
    model = small_mst(5, 5, pixel_chunk_size=7)
    with torch.no_grad():
        for head in model.encoder.heads:
            head.weight.normal_(std=0.01)
    model.eval()
    context, x = torch.rand(1, 5, 9, 11), torch.rand(1, 5, 9, 11)
    parameters = model.generate_parameters(context)
    prediction = model.predictor(context)
    torch.testing.assert_close(prediction(x), model.apply_parameters(x, parameters))
    expected = model(x)
    model.pixel_chunk_size = 10000
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("kwargs", [
    {"grid_range": (1, 0)}, {"grid_range": (0, float("nan"))},
    {"grid_size": 0}, {"spline_order": 0}, {"in_dim": 0},
])
def test_invalid_layer_settings(kwargs):
    params = dict(in_dim=3, out_dim=3)
    params.update(kwargs)
    with pytest.raises(ValueError):
        SharedSplineLayer(**params)


def test_non_ntire_channels_require_explicit_colorimetry_opt_out():
    with pytest.raises(ValueError, match="31 bands"):
        SharedCmKANPipeline(torch.nn.Identity(), spectral_channels=7)
    with pytest.raises(ValueError, match="perceptual_weight"):
        SharedCmKANPipeline(torch.nn.Identity(), spectral_channels=7,
                            use_ntire_colorimetry=False)
    pipeline = SharedCmKANPipeline(torch.nn.Identity(), spectral_channels=7,
                                  use_ntire_colorimetry=False, perceptual_weight=0)
    src, target = torch.rand(1, 7, 8, 8), torch.rand(1, 7, 8, 8)
    losses = pipeline.restoration_losses(src, target)
    assert torch.isfinite(losses["loss"])
    assert set(pipeline.quality_metrics(src, target)) == {"psnr", "sam"}


@pytest.mark.parametrize("channels", [7, 31])
def test_configured_ntire_training_and_checkpoint(tmp_path, channels):
    root = tmp_path / "dataset"
    crop_size = 32 if channels == 31 else 8
    for split in ("train", "test"):
        for domain in ("source", "target"):
            folder = root / split / domain
            folder.mkdir(parents=True)
            cube = np.random.default_rng(7).uniform(
                0.1, 0.9, (crop_size + 4, crop_size + 8, channels),
            ).astype("float32")
            np.save(folder / "sample.npy", cube * (0.9 if domain == "target" else 1))
    options = {
        "dataset_root": root.as_posix(), "spectral_channels": channels,
        "accelerator": "cpu", "devices": 1, "epochs": 1,
        "save_dir": (tmp_path / "experiments").as_posix(),
        "data.params.num_workers": 0, "data.params.pin_memory": "false",
        "data.params.train.crop_size": crop_size, "data.params.val.crop_size": crop_size,
    }
    if channels != 31:
        options.update({"pipeline.params.use_ntire_colorimetry": "false",
                        "pipeline.params.perceptual_weight": 0})
    config = configuration(**options)
    dm = DataSelector.select(config.data)
    dm.setup("fit")
    x, target = next(iter(dm.train_dataloader()))
    model = ModelSelector.select(config.model)
    pipeline = PipelineSelector.select(model, config.pipeline)
    assert isinstance(pipeline, SharedCmKANPipeline)
    assert model.layers.cmkan.in_channels == channels
    assert x.shape == target.shape == (1, channels, crop_size, crop_size)
    torch.testing.assert_close(pipeline(x), x)
    train_default(config)
    checkpoint = tmp_path / "experiments" / config.experiment / "logs" / "checkpoints" / "last.ckpt"
    assert checkpoint.is_file()
    state = torch.load(checkpoint, weights_only=False)
    assert state["global_step"] == 1
    assert all(torch.isfinite(v).all() for v in state["state_dict"].values())
    assert not torch.equal(state["state_dict"]["model.layers.cmkan.spline_layers.0.base_parameters"],
                           model.layers.cmkan.spline_layers[0].base_parameters)
