from pathlib import Path

import lightning as L
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from skimage.color import deltaE_ciede2000

from flows.core.selector import ModelSelector, PipelineSelector
from flows.ml.layers.mst import MSAB
from flows.ml.models.ggpir import ParallelHyperspectralUNets
from flows.ml.models.ggpir.usgs_unet import GaussianInjection
from flows.ml.layers.spectral_color import NTIREColorimetry
from flows.ml.models.flow import Flow
from flows.ml.pipelines.usgs_unet import USGSUNetPipeline
from flows.cli.train import train_default


@pytest.fixture(autouse=True)
def small_cpu_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(threads)


def make_model(cascades=2, use_hypernet=True):
    return ParallelHyperspectralUNets(
        num_cascades=cascades, cm_base_channels=4,
        hyper_channels=4, hyper_base_channels=4, num_gaussians=2,
        use_hypernet=use_hypernet,
    )


class ModelFlow(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src):
        return {"res": self.model(src)}


def make_pipeline(**kwargs):
    return USGSUNetPipeline(ModelFlow(make_model()), **kwargs)


@pytest.mark.parametrize("shape", [(2, 31, 17, 19), (1, 31, 8, 8), (1, 31, 1, 1)])
def test_msab_parameter_cascades_preserve_shape_and_identity(shape):
    model = make_model()
    assert sum(isinstance(m, MSAB) for m in model.hyper_stages.modules()) == 10
    assert not any(isinstance(m, MSAB) for m in model.cm_stages.modules())
    for stage in model.hyper_stages:
        assert not any(isinstance(m, MSAB) for m in stage.encoder0.modules())
        assert not any(isinstance(m, MSAB) for m in stage.decoder0.modules())
        assert any(isinstance(m, MSAB) for m in stage.bottleneck.modules())
    assert all(isinstance(m, GaussianInjection) for m in model.injections)
    x = torch.rand(shape)
    y, stages = model(x, return_intermediate=True)
    assert len(stages) == 2
    for stage in stages:
        torch.testing.assert_close(stage, x, atol=1e-6, rtol=1e-6)
    assert y.shape == x.shape


def test_gaussian_full_attention_baseline_remains_available():
    model = ParallelHyperspectralUNets(
        num_cascades=3, cm_base_channels=4, hyper_channels=4,
        hyper_base_channels=4, use_hypernet=True,
        hyper_full_resolution_msab=True, num_gaussians=2,
    )
    assert all(isinstance(m, GaussianInjection) for m in model.injections)
    assert sum(isinstance(m, MSAB) for m in model.hyper_stages.modules()) == 21
    x = torch.rand(1, 31, 9, 11)
    torch.testing.assert_close(model(x), x)


@pytest.mark.parametrize("cascades", [1, 3, 4])
@pytest.mark.parametrize("use_hypernet", [False, True])
def test_optional_hypernet_and_configurable_cascade_count(cascades, use_hypernet):
    model = make_model(cascades, use_hypernet)
    assert len(model.cm_stages) == cascades
    assert len(model.hyper_stages) == (cascades if use_hypernet else 0)
    assert len(model.injections) == (cascades if use_hypernet else 0)
    if not use_hypernet:
        assert model.hyper_stem is None
        assert not any("hyper" in key or "injections" in key for key in model.state_dict())
        assert not any(isinstance(m, MSAB) for m in model.modules())
    x = torch.rand(1, 31, 9, 11)
    y, stages = model(x, return_intermediate=True)
    assert len(stages) == cascades
    torch.testing.assert_close(y, x)
    # Verify all CM cascades execute even without a hypernetwork.
    with torch.no_grad():
        for stage in model.cm_stages:
            stage.output.bias.fill_(0.01)
    y, stages = model(x, return_intermediate=True)
    for index, stage in enumerate(stages):
        torch.testing.assert_close(stage, x + 0.01 * (index + 1))
    (y - x * 0.8).abs().mean().backward()
    assert all(stage.output.weight.grad is not None for stage in model.cm_stages)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_default_constructor_disables_hypernet_and_has_three_cascades():
    model = ParallelHyperspectralUNets(cm_base_channels=4)
    assert len(model.cm_stages) == 3
    assert model.hyper_stem is None
    assert len(model.hyper_stages) == len(model.injections) == 0
    with pytest.raises(ValueError, match="num_cascades"):
        make_model(cascades=0)
    with pytest.raises(ValueError, match="boolean"):
        make_model(use_hypernet="false")


def test_masking_and_held_out_objective():
    pipeline = make_pipeline(mask_ratio=0.5)
    target = torch.rand(3, 31, 9, 11) + 0.1
    rng = torch.get_rng_state()
    masked, mask = pipeline.mask_channels(target, deterministic=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(mask, pipeline.mask_channels(target, deterministic=True)[1])
    assert mask.flatten(1).sum(dim=1).tolist() == [16, 16, 16]
    assert torch.count_nonzero(masked.masked_select(mask.expand_as(masked))) == 0
    torch.testing.assert_close(masked.masked_select(~mask.expand_as(masked)),
                               target.masked_select(~mask.expand_as(target)))
    pred = target.clone().masked_fill(~mask, 100.0).requires_grad_()
    loss = pipeline.masked_mae(pred, target, mask)
    assert loss.item() == 0
    pred = target.clone().masked_fill(mask, 0).requires_grad_()
    pipeline.masked_mae(pred, target, mask).backward()
    assert torch.count_nonzero(pred.grad.masked_select(~mask.expand_as(pred))) == 0
    assert torch.count_nonzero(pred.grad.masked_select(mask.expand_as(pred))) > 0
    assert not torch.equal(pipeline.mask_channels(target)[1], pipeline.mask_channels(target)[1])


def test_masked_reconstruction_learns_and_reaches_parameter_branch():
    pipeline = make_pipeline(lr=1e-2)
    target = torch.full((1, 31, 9, 11), 0.4)
    masked, mask = pipeline.mask_channels(target, deterministic=True)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=1e-2)
    initial = pipeline.masked_mae(pipeline(masked), target, mask).item()
    for _ in range(15):
        optimizer.zero_grad()
        loss = pipeline.masked_mae(pipeline(masked), target, mask)
        loss.backward()
        optimizer.step()
    assert pipeline.masked_mae(pipeline(masked), target, mask).item() < initial * 0.5
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in pipeline.model.model.hyper_stages.parameters())


def test_pretraining_uses_targets_and_finetuning_uses_sources(monkeypatch):
    pipeline = make_pipeline()
    src = torch.full((1, 31, 9, 11), 0.2)
    target = torch.full_like(src, 0.8)
    inputs = []

    def record_input(x):
        inputs.append(x)
        return x

    monkeypatch.setattr(pipeline, "forward", record_input)
    monkeypatch.setattr(pipeline, "log", lambda *args, **kwargs: None)
    masked_loss = pipeline.training_step((src, target), 0)["loss"]
    assert inputs[0].eq(0).any() and inputs[0].eq(0.8).any()
    assert not inputs[0].eq(0.2).any()
    torch.testing.assert_close(masked_loss, torch.tensor(0.8))
    monkeypatch.setattr(USGSUNetPipeline, "pretraining", False)
    paired_loss = pipeline.training_step((src, target), 0)["loss"]
    assert inputs[1] is src
    torch.testing.assert_close(paired_loss, pipeline.restoration_losses(src, target)["loss"])


@pytest.mark.parametrize("value", [0.0, 1e-7, 0.4])
def test_restoration_loss_has_finite_gradients_for_dark_and_matched_spectra(value):
    pipeline = make_pipeline(pretrain_epochs=0)
    target = torch.full((2, 31, 8, 9), value)
    pred = target.clone().requires_grad_()
    terms = pipeline.restoration_losses(pred, target)
    terms["loss"].backward()
    assert torch.isfinite(terms["loss"])
    assert torch.isfinite(pred.grad).all()
    assert terms["loss"].item() < 1e-6


def test_calibrated_color_projection():
    response = torch.ones(3, 31) / 31
    pipeline = make_pipeline(color_weight=0.2, spectral_response=response.tolist())
    pred = torch.ones(1, 31, 2, 2, requires_grad=True)
    target = torch.zeros_like(pred)
    terms = pipeline.restoration_losses(pred, target)
    torch.testing.assert_close(terms["color"], torch.tensor(1.0))
    terms["loss"].backward()
    assert torch.isfinite(pred.grad).all()
    with pytest.raises(ValueError, match="channel count"):
        pipeline.restoration_losses(torch.ones(1, 30, 2, 2), torch.zeros(1, 30, 2, 2))


@pytest.mark.parametrize("kwargs", [
    {"pretrain_epochs": -1}, {"mask_ratio": 0}, {"mask_ratio": 1},
    {"lr": 0}, {"spectral_weight": -1}, {"color_weight": 1},
    {"spectral_response": [[1, 2]]}, {"spectral_response": [[0, 0]] * 3},
    {"spectral_response": [[float("nan"), 1]] * 3},
    {"perceptual_weight": -1},
])
def test_invalid_pipeline_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make_pipeline(**kwargs)


def test_default_configuration_uses_ntire_msab_model_and_pipeline():
    config_dir = str(Path(__file__).resolve().parents[1] / ".configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config")
    assert cfg.data.name == "ntire"
    assert cfg.pipeline.name == "usgs_unet"
    assert cfg.pipeline.params.pretrain_epochs < cfg.epochs
    assert not cfg.swa
    assert cfg.model.params.layers.pyramid.params.num_cascades == 3
    assert cfg.model.params.layers.pyramid.params.use_hypernet is False
    assert cfg.pipeline.params.perceptual_weight > 0
    model = ModelSelector.select(cfg.model)
    pipeline = PipelineSelector.select(model, cfg.pipeline)
    assert isinstance(model, Flow)
    assert isinstance(pipeline, USGSUNetPipeline)
    x = torch.rand(1, 31, 9, 11)
    torch.testing.assert_close(pipeline(x), x, atol=1e-6, rtol=1e-6)


class RecordPhase(Callback):
    def __init__(self, stop=False):
        self.phases = []
        self.stop = stop

    def on_train_epoch_start(self, trainer, pl_module):
        self.phases.append(pl_module.pretraining)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.stop:
            trainer.should_stop = True


def trainer(tmp_path, callbacks):
    return L.Trainer(
        accelerator="cpu", devices=1, max_epochs=2,
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=tmp_path, callbacks=callbacks, num_sanity_val_steps=0,
    )


def test_training_transitions_and_resumes_with_identity_initialization_intact(tmp_path):
    src = torch.rand(2, 31, 9, 11)
    loader = DataLoader(TensorDataset(src, src * 0.8), batch_size=1)
    pipeline = make_pipeline(pretrain_epochs=1)
    first_phase = RecordPhase(stop=True)
    checkpoint = ModelCheckpoint(monitor="val_mrae", save_last=True)
    first = trainer(tmp_path, [first_phase, checkpoint])
    # setup must preserve zero Gaussian and CM output heads.
    pipeline.trainer = first
    before = {k: v.clone() for k, v in pipeline.state_dict().items()}
    pipeline.setup("fit")
    for k, v in pipeline.state_dict().items():
        torch.testing.assert_close(v, before[k])
    first.fit(pipeline, loader, loader)
    assert first_phase.phases == [True]
    assert "val_mrae" in first.callback_metrics
    assert "val_masked_mae" in first.callback_metrics
    assert "train_psnr_step" in first.progress_bar_metrics
    assert "train_de_step" in first.progress_bar_metrics
    assert "val_psnr" in first.progress_bar_metrics
    assert "val_de" in first.progress_bar_metrics
    assert Path(checkpoint.last_model_path).is_file()

    second_phase = RecordPhase()
    second = trainer(tmp_path, [second_phase, ModelCheckpoint(monitor="val_mrae", save_last=True)])
    resumed = make_pipeline(pretrain_epochs=1)
    second.fit(resumed, loader, loader, ckpt_path=checkpoint.last_model_path)
    assert second_phase.phases == [False]
    assert second.global_step == 4
    assert "mrae" in second.callback_metrics
    assert "train_psnr_step" in second.progress_bar_metrics
    assert "train_de_step" in second.progress_bar_metrics
    assert "val_psnr" in second.progress_bar_metrics
    assert "val_de" in second.progress_bar_metrics
    assert torch.isfinite(second.callback_metrics["val_mrae"])
    assert second.test(resumed, loader, verbose=False)[0]["test_mrae"] >= 0
    prediction = resumed.predict_step((src, src, ["a.npy", "b.npy"]), 0)
    assert prediction["image"].shape == src.shape
    assert prediction["filenames"] == ["a.npy", "b.npy"]


def test_pretraining_requires_remaining_finetuning_epochs(tmp_path):
    pipeline = make_pipeline(pretrain_epochs=2)
    pipeline.trainer = trainer(tmp_path, [])
    with pytest.raises(ValueError, match="epochs must exceed"):
        pipeline.setup("fit")


def test_bfloat16_autocast_has_finite_backward():
    pipeline = make_pipeline()
    src = torch.rand(1, 31, 9, 11)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = pipeline.restoration_losses(pipeline(src), src * 0.8)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in pipeline.parameters() if p.grad is not None)


@pytest.mark.parametrize("use_hypernet", [False, True])
def test_ntire_training_entrypoint_writes_checkpoints_and_previews(tmp_path, use_hypernet):
    config_dir = str(Path(__file__).resolve().parents[1] / ".configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config")
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    cube = np.random.default_rng(0).random((12, 12, 31), dtype=np.float32)
    np.save(source / "sample.npy", cube)
    np.save(target / "sample.npy", cube * 0.8)
    for split in ("train", "val", "test"):
        cfg.data.params[split].source = str(source)
        cfg.data.params[split].target = str(target)
        cfg.data.params[split].batch_size = 1
    cfg.data.params.train.crop_size = 8
    cfg.data.params.num_workers = 0
    cfg.epochs = 2
    cfg.pipeline.params.pretrain_epochs = 1
    cfg.accelerator = "cpu"
    cfg.devices = 1
    cfg.save_dir = str(tmp_path)
    cfg.experiment = "ntire-smoke"
    cfg.resume = False
    cfg.visualize_freq = 1
    params = cfg.model.params.layers.pyramid.params
    params.num_cascades = 1
    params.cm_base_channels = 4
    params.hyper_channels = 4
    params.hyper_base_channels = 4
    params.num_gaussians = 2
    params.use_hypernet = use_hypernet
    train_default(cfg)
    logs = tmp_path / cfg.experiment / "logs"
    assert (logs / "checkpoints" / "last.ckpt").is_file()
    assert (logs / "figures" / "reconst_0.png").is_file()
    assert (logs / "figures" / "reconst_1.png").is_file()
    metrics = (logs / "metrics.csv").read_text()
    for metric in ("val_mrae", "masked_mae", "train_psnr", "train_de",
                   "val_psnr", "val_de", "perceptual"):
        assert metric in metrics


@pytest.mark.parametrize("value,lightness", [(0.0, 0.0), (1.0, 100.0), (0.18, 49.4961076)])
def test_cie_conversion_reference_neutrals(value, lightness):
    color = NTIREColorimetry()
    lab = color(torch.full((1, 31, 2, 3), value))
    expected = torch.tensor([lightness, 0.0, 0.0])[None, :, None, None].expand_as(lab)
    torch.testing.assert_close(lab, expected, atol=5e-5, rtol=1e-6)
    torch.testing.assert_close(color.response[1].sum(), torch.tensor(1.0))
    # 600 nm CIE 1931 observer chromatic ratios, independent of D65 intensity.
    torch.testing.assert_close(color.response[:, 20] / color.response[1, 20],
                               torch.tensor([1.0622 / 0.631, 1.0, 0.0008 / 0.631]))
    with pytest.raises(ValueError, match="31 ordered bands"):
        color(torch.zeros(1, 3, 2, 3))


@pytest.mark.parametrize("value", [-0.1, 0.0, 0.001, 0.5, 1.2])
def test_perceptual_loss_gradient_is_finite_and_nonzero(value):
    pipeline = make_pipeline(
        mrae_weight=0, spectral_weight=0, spectral_gradient_weight=0,
        perceptual_weight=1,
    )
    pred = torch.full((1, 31, 3, 5), value, requires_grad=True)
    target = torch.full_like(pred, 0.3)
    perceptual = pipeline.restoration_losses(pred, target)["perceptual"]
    perceptual.backward()
    assert torch.isfinite(perceptual)
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0
    assert torch.count_nonzero(pred.grad.flatten(2).sum(dim=2)) == 31


def test_quality_metrics_match_independent_ciede2000_and_psnr():
    pipeline = make_pipeline()
    pred, target = torch.rand(2, 31, 4, 5), torch.rand(2, 31, 4, 5)
    actual = pipeline.quality_metrics(pred, target)
    lab_pred = pipeline.colorimetry(pred).permute(0, 2, 3, 1).numpy().astype(np.float64)
    lab_target = pipeline.colorimetry(target).permute(0, 2, 3, 1).numpy().astype(np.float64)
    expected_de = deltaE_ciede2000(lab_pred, lab_target).mean()
    assert actual["de"].item() == pytest.approx(expected_de, rel=1e-5)
    expected_psnr = (-10 * np.log10(
        np.mean((pred.numpy() - target.numpy()) ** 2, axis=(1, 2, 3)))).mean()
    assert actual["psnr"].item() == pytest.approx(expected_psnr, rel=1e-5)
    matched = pipeline.quality_metrics(target, target)
    assert matched["de"] == 0
    assert torch.isfinite(matched["psnr"])
