from pathlib import Path

import lightning as L
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flows.core.selector import ModelSelector, PipelineSelector
from flows.ml.layers.mst import MSAB
from flows.ml.models.ggpir import ParallelHyperspectralUNets
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


def make_model(cascades=2):
    return ParallelHyperspectralUNets(
        num_cascades=cascades, cm_base_channels=4,
        hyper_channels=4, hyper_base_channels=4, num_gaussians=2,
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
    assert sum(isinstance(m, MSAB) for m in model.hyper_stages.modules()) == 14
    assert not any(isinstance(m, MSAB) for m in model.cm_stages.modules())
    x = torch.rand(shape)
    y, stages = model(x, return_intermediate=True)
    assert len(stages) == 2
    for stage in stages:
        torch.testing.assert_close(stage, x, atol=1e-6, rtol=1e-6)
    assert y.shape == x.shape


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
    # setup must not reset zero Gaussian amplitudes, centers, or CM output heads.
    pipeline.trainer = first
    before = {k: v.clone() for k, v in pipeline.state_dict().items()}
    pipeline.setup("fit")
    for k, v in pipeline.state_dict().items():
        torch.testing.assert_close(v, before[k])
    first.fit(pipeline, loader, loader)
    assert first_phase.phases == [True]
    assert "val_mrae" in first.callback_metrics
    assert "val_masked_mae" in first.callback_metrics
    assert Path(checkpoint.last_model_path).is_file()

    second_phase = RecordPhase()
    second = trainer(tmp_path, [second_phase, ModelCheckpoint(monitor="val_mrae", save_last=True)])
    resumed = make_pipeline(pretrain_epochs=1)
    second.fit(resumed, loader, loader, ckpt_path=checkpoint.last_model_path)
    assert second_phase.phases == [False]
    assert second.global_step == 4
    assert "mrae" in second.callback_metrics
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


def test_ntire_training_entrypoint_writes_checkpoints_and_previews(tmp_path):
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
    params = cfg.model.params.layers.pyramid.params
    params.num_cascades = 1
    params.cm_base_channels = 4
    params.hyper_channels = 4
    params.hyper_base_channels = 4
    params.num_gaussians = 2
    train_default(cfg)
    logs = tmp_path / cfg.experiment / "logs"
    assert (logs / "checkpoints" / "last.ckpt").is_file()
    assert (logs / "figures" / "reconst_0.png").is_file()
    assert (logs / "figures" / "reconst_1.png").is_file()
    metrics = (logs / "metrics.csv").read_text()
    assert "val_mrae" in metrics and "masked_mae" in metrics
