from pathlib import Path
from types import SimpleNamespace

import lightning as L
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional.image import structural_similarity_index_measure

from flows.core import ModelSelector, PipelineSelector
from flows.cli.train import train_default
from flows.ml.models.ggpir.spectral_encoder_decoder import (
    ChannelLayerNorm, ReconstructionHead, SpectralAttentionBlock, SpectralEncoderDecoder,
)
from flows.ml.pipelines.spectral import SpectralExperimentPipeline


@pytest.fixture(autouse=True)
def cpu_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(threads)


def network(head="pretrain"):
    return SpectralEncoderDecoder(
        head=head, channels=(4, 8, 16, 24), depths=(1, 1, 1),
        heads=(2, 4, 6), reconstruction_channels=(12, 8, 4),
    )


def pipeline(phase="pretrain", **kwargs):
    return SpectralExperimentPipeline(network(phase), phase=phase, **kwargs)


def loader():
    src = torch.rand(4, 31, 12, 16)
    return DataLoader(TensorDataset(src, src * 0.8), batch_size=2)


def trainer(tmp_path, callbacks=(), **kwargs):
    return L.Trainer(
        default_root_dir=tmp_path, accelerator="cpu", devices=1,
        max_epochs=2, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        callbacks=list(callbacks), **kwargs,
    )


def export_encoder(tmp_path):
    module = pipeline()
    run = trainer(tmp_path, enable_checkpointing=False)
    run.fit(module, loader(), loader())
    assert {"train_sam_step", "val_sam"} <= run.progress_bar_metrics.keys()
    return module, tmp_path / "checkpoints" / "encoder.pt"


@pytest.mark.parametrize("head", ["pretrain", "restoration"])
@pytest.mark.parametrize("shape", [(2, 31, 17, 19), (1, 31, 8, 8), (1, 31, 1, 1)])
def test_encoder_heads_shape_and_finite_backward(head, shape):
    model = network(head)
    x = torch.rand(shape)
    mask = torch.zeros((*x.shape[:2], 1, 1), dtype=torch.bool)
    mask[:, ::2] = True
    output = model(x.masked_fill(mask, 0), mask)
    assert output.shape == x.shape
    (output - x).abs().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert model.encoder.stem.weight.grad.abs().sum() > 0
    assert not any(isinstance(m, (nn.GroupNorm, nn.BatchNorm2d))
                   for m in model.modules())


def test_per_pixel_layernorm_is_independent_of_other_pixels():
    norm = ChannelLayerNorm(4)
    x = torch.rand(1, 4, 3, 5)
    changed = x.clone()
    changed[:, :, 2, 4] = torch.arange(4).float() * 100
    torch.testing.assert_close(norm(x)[..., 0, 0], norm(changed)[..., 0, 0])


def test_simple_head_is_bottleneck_only_and_restoration_has_attention():
    model = network()
    features = model.encoder(torch.rand(1, 31, 17, 19))
    expected = model.decoder(features)
    changed = [torch.randn_like(tensor) for tensor in features[:-1]] + features[-1:]
    torch.testing.assert_close(model.decoder(changed), expected)
    assert isinstance(model.decoder, ReconstructionHead)
    assert not any(isinstance(m, SpectralAttentionBlock) for m in model.decoder.modules())
    restoration = network("restoration")
    assert sum(isinstance(m, SpectralAttentionBlock)
               for m in restoration.decoder.modules()) == 3
    assert not any(isinstance(m, SpectralAttentionBlock)
                   for m in restoration.decoder.decoder0.modules())
    assert restoration.decoder.output.weight.count_nonzero() > 0


def test_mask_embedding_distinguishes_hidden_from_dark_and_all_visible_is_compatible():
    model = network().eval()
    x = torch.zeros(1, 31, 8, 8)
    visible = torch.zeros(1, 31, 1, 1, dtype=torch.bool)
    hidden = visible.clone()
    hidden[:, ::2] = True
    torch.testing.assert_close(model(x), model(x, visible))
    assert not torch.allclose(model(x), model(x, hidden))
    with pytest.raises(ValueError, match="band_mask"):
        model(x, hidden.float())


@pytest.mark.parametrize("head", ["pretrain", "restoration"])
def test_cpu_bfloat16_mixed_precision_backward(head):
    model = network(head)
    target = torch.rand(1, 31, 12, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prediction = model(target)
        loss = (prediction.float() - target).abs().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_pretraining_alternates_source_target_self_reconstruction(monkeypatch):
    module = pipeline(visible_weight=0.1)
    src, target = torch.full((1, 31, 8, 8), 0.2), torch.full((1, 31, 8, 8), 0.8)
    captured = []

    def forward(x, y=None, band_mask=None):
        captured.append((x, band_mask))
        return torch.zeros_like(x)

    monkeypatch.setattr(module, "forward", forward)
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    quality_targets = []
    monkeypatch.setattr(module, "log_quality",
                        lambda pred, reference, prefix: quality_targets.append(reference))
    for epoch in (0, 1, 4):
        module._trainer = SimpleNamespace(current_epoch=epoch)
        for batch_idx in range(3):
            reference = src if (batch_idx + epoch) % 2 == 0 else target
            result = module.training_step((src, target), batch_idx)
            masked, mask = captured[-1]
            assert mask.sum() == 16
            assert masked[mask.expand_as(masked)].count_nonzero() == 0
            torch.testing.assert_close(
                masked[~mask.expand_as(masked)], reference[~mask.expand_as(masked)],
            )
            torch.testing.assert_close(result["loss"], reference.mean() * 1.1)
            assert quality_targets[-1] is reference
    first = module.validation_step((src, target), 0)
    torch.testing.assert_close(first["loss"], torch.tensor(0.88))
    assert quality_targets[-1] is target
    first_mask = captured[-1][1]
    second = module.validation_step((src, target), 0)
    assert torch.equal(first_mask, captured[-1][1])
    assert "preview_source" in first
    torch.testing.assert_close(first["loss"], second["loss"])
    module.validation_step((src, target), 1)
    assert not torch.equal(first_mask, captured[-1][1])


def test_pretraining_export_and_strict_encoder_only_transfer(tmp_path):
    pretrained, path = export_encoder(tmp_path)
    assert path.is_file()
    artifact = torch.load(path, weights_only=True)
    assert artifact["format"] == "flow-hsi.spectral-encoder.v1"
    assert artifact["architecture"] == pretrained.network.encoder.architecture()
    assert not any("decoder" in key for key in artifact["encoder_state_dict"])
    restore = pipeline("restoration", encoder_checkpoint=str(path))
    before = {key: value.clone() for key, value in restore.network.decoder.state_dict().items()}
    restore.load_encoder(str(path))
    for key, value in restore.network.encoder.state_dict().items():
        torch.testing.assert_close(value, artifact["encoder_state_dict"][key])
    for key, value in restore.network.decoder.state_dict().items():
        torch.testing.assert_close(value, before[key])
    artifact["architecture"]["channels"][0] = 999
    torch.save(artifact, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="Incompatible"):
        restore.load_encoder(str(tmp_path / "wrong.pt"))
    with pytest.raises(FileNotFoundError):
        restore.load_encoder(str(tmp_path / "missing.pt"))


@pytest.mark.parametrize("phase", ["pretrain", "restoration"])
@pytest.mark.parametrize("size", [1, 4, 8, 16])
def test_ssim_loss_composition_and_mixed_precision_gradients(phase, size):
    module = pipeline(phase, ssim_weight=0.15)
    target = torch.rand(1, 31, size, size)
    pred = (target * 0.8 + 0.05).bfloat16().requires_grad_()
    mask = torch.zeros(1, 31, 1, 1, dtype=torch.bool)
    mask[:, ::2] = True
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        if phase == "pretrain":
            terms = module.reconstruction_losses(pred, target, mask)
            base = terms["masked_mae"] + module.visible_weight * terms["visible_mae"]
        else:
            terms = module.restoration_losses(pred, target)
            module.ssim_weight = 0
            base = module.restoration_losses(pred, target)["loss"]
            module.ssim_weight = 0.15
    assert terms["ssim_loss"].dtype == torch.float32
    torch.testing.assert_close(terms["loss"], base + 0.15 * terms["ssim_loss"])
    terms["loss"].backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0
    torch.testing.assert_close(module.ssim_loss(target, target), torch.tensor(0.0))
    if size == 16:
        expected = 1 - structural_similarity_index_measure(
            pred.detach().float(), target, data_range=1.0,
        )
        torch.testing.assert_close(terms["ssim_loss"], expected)


def test_best_export_survives_worse_validation_and_resume(tmp_path):
    module = pipeline()
    module._trainer = SimpleNamespace(
        sanity_checking=True, is_global_zero=True, callback_metrics={
            "val_masked_mae": torch.tensor(0.2),
        }, log_dir=str(tmp_path), current_epoch=0, global_step=4,
    )
    path = tmp_path / "checkpoints" / "encoder.pt"
    module.on_validation_end()
    assert not path.exists()
    module._trainer.sanity_checking = False
    checkpoint = {}
    # Lightning can save training checkpoints before the module's export hook.
    module.on_save_checkpoint(checkpoint)
    module.on_validation_end()
    before = path.read_bytes()
    resumed = pipeline()
    resumed.on_load_checkpoint(checkpoint)
    resumed._trainer = module._trainer
    resumed._trainer.current_epoch = 6
    resumed._trainer.callback_metrics["val_masked_mae"] = torch.tensor(0.3)
    resumed.on_validation_end()
    assert path.read_bytes() == before
    assert resumed.pretraining
    resumed._trainer.callback_metrics["val_masked_mae"] = torch.tensor(0.1)
    resumed.on_validation_end()
    assert torch.load(path, weights_only=True)["val_masked_mae"] == pytest.approx(0.1)
    resumed._trainer.callback_metrics["val_masked_mae"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        resumed.on_validation_end()
    assert not (tmp_path / "checkpoints" / "encoder.pt.tmp").exists()


def test_full_checkpoint_can_reconstruct_pipeline_with_separate_model(tmp_path):
    module = pipeline()
    run = trainer(tmp_path, enable_checkpointing=False)
    run.fit(module, loader(), loader())
    run.save_checkpoint(tmp_path / "full.ckpt")
    loaded = SpectralExperimentPipeline.load_from_checkpoint(
        tmp_path / "full.ckpt", model=network(),
    )
    assert loaded.phase == "pretrain"
    assert loaded._resuming
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])


def test_resume_rejects_mismatched_phase_or_freeze_schedule(tmp_path):
    module = pipeline()
    module._trainer = SimpleNamespace(callback_metrics={})
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="matching spectral"):
        pipeline("restoration").on_load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="matching spectral"):
        pipeline(freeze_encoder_steps=1).on_load_checkpoint(checkpoint)


class FreezeRecord(Callback):
    def __init__(self):
        self.records = []
        self.head_before = None
        self.encoder_before = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Module hooks run after callback hooks; check after the actual update.
        if self.head_before is None:
            self.head_before = pl_module.network.decoder.output.weight.detach().clone()
            self.encoder_before = pl_module.network.encoder.stem.weight.detach().clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        encoder = pl_module.network.encoder
        self.records.append({
            "step": trainer.global_step,
            "frozen": not encoder.stem.weight.requires_grad,
            "encoder_changed": not torch.equal(self.encoder_before, encoder.stem.weight),
            "decoder_changed": not torch.equal(self.head_before,
                                               pl_module.network.decoder.output.weight),
            "encoder_has_grad": encoder.stem.weight.grad is not None,
        })


def test_head_only_updates_then_whole_network_and_optimizer_groups(tmp_path):
    pretrained, path = export_encoder(tmp_path / "pretrain")
    module = pipeline("restoration", encoder_checkpoint=str(path), freeze_encoder_steps=1)
    record = FreezeRecord()
    run = trainer(tmp_path / "restore", callbacks=[record], enable_checkpointing=False)
    run.fit(module, loader(), loader())
    first, second = record.records[:2]
    assert first["frozen"] and not first["encoder_changed"]
    assert first["decoder_changed"] and not first["encoder_has_grad"]
    assert not second["frozen"] and second["encoder_changed"] and second["encoder_has_grad"]
    optimizer = run.optimizers[0]
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["initial_lr"] == 2e-5
    assert optimizer.param_groups[1]["initial_lr"] == 2e-4
    assert optimizer.state[module.network.decoder.output.weight]["step"] == 4
    assert {"train_psnr_step", "train_de_step", "train_sam_step",
            "val_psnr", "val_de", "val_sam"} <= run.progress_bar_metrics.keys()


def test_freeze_is_capped_at_one_epoch(tmp_path):
    module = pipeline("restoration", allow_random_encoder=True, freeze_encoder_steps=500)
    record = FreezeRecord()
    run = trainer(tmp_path, callbacks=[record], enable_checkpointing=False)
    run.fit(module, loader(), loader())
    assert [entry["frozen"] for entry in record.records] == [True, True, False, False]


def test_freeze_counts_optimizer_updates_with_accumulation(tmp_path):
    module = pipeline("restoration", allow_random_encoder=True, freeze_encoder_steps=1)
    record = FreezeRecord()
    run = trainer(tmp_path, callbacks=[record], enable_checkpointing=False,
                  accumulate_grad_batches=2)
    run.fit(module, loader(), loader())
    assert [entry["frozen"] for entry in record.records] == [True, True, False, False]
    assert run.global_step == 2


def test_resume_restores_frozen_phase_and_does_not_reload_encoder_artifact(tmp_path):
    _, path = export_encoder(tmp_path / "pretrain")
    artifact = str(path)
    module = pipeline("restoration", encoder_checkpoint=artifact, freeze_encoder_steps=2)
    checkpoint = ModelCheckpoint(monitor="val_mrae", save_last=True)
    record = FreezeRecord()

    class SaveMidEpoch(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            trainer.save_checkpoint(tmp_path / "mid_epoch.ckpt")

    first = trainer(tmp_path / "restore", callbacks=[
        checkpoint, record, SaveMidEpoch(),
    ], max_steps=1)
    first.fit(module, loader(), loader())
    assert record.records[0]["frozen"]
    path.unlink()
    resumed = pipeline("restoration", encoder_checkpoint=artifact, freeze_encoder_steps=2)
    resumed_record = FreezeRecord()
    second = trainer(tmp_path / "restore", callbacks=[
        resumed_record, ModelCheckpoint(monitor="val_mrae", save_last=True),
    ])
    second.fit(resumed, loader(), loader(), ckpt_path=tmp_path / "mid_epoch.ckpt")
    assert resumed._resuming
    assert resumed_record.records[0]["frozen"]
    assert any(not item["frozen"] for item in resumed_record.records)
    assert second.global_step >= 3


def test_restore_without_encoder_does_not_silently_train_random(tmp_path):
    module = pipeline("restoration")
    with pytest.raises(ValueError, match="requires encoder_checkpoint"):
        trainer(tmp_path, enable_checkpointing=False).fit(module, loader())


@pytest.mark.parametrize("phase,epochs", [("pretrain", 5), ("restore", 20)])
def test_experiment_config_and_factory(phase, epochs):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / ".configs"), version_base="1.3",
    ):
        cfg = compose(config_name=phase)
    assert cfg.epochs == epochs
    assert cfg.pipeline.params.ssim_weight == 0.15
    module = PipelineSelector.select(ModelSelector.select(cfg.model), cfg.pipeline)
    assert isinstance(module, SpectralExperimentPipeline)
    assert module.checkpoint_monitor == ("val_masked_mae" if phase == "pretrain" else "val_mrae")
    assert len(module.network.encoder.stages) == 3
    x = torch.rand(1, 31, 17, 19)
    assert module(x).shape == x.shape
    assert module.model(src=x)["res"].shape == x.shape


def test_two_experiments_through_real_ntire_entrypoint(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    cube = np.random.default_rng(1).random((16, 16, 31), dtype=np.float32)
    np.save(source / "sample.npy", cube)
    np.save(target / "sample.npy", cube * 0.8)
    configs = {}
    for phase in ("pretrain", "restore"):
        with initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parents[1] / ".configs"), version_base="1.3",
        ):
            cfg = compose(config_name=phase)
        cfg.epochs = 2
        cfg.save_dir = str(tmp_path)
        cfg.accelerator = "cpu"
        cfg.devices = 1
        cfg.resume = False
        cfg.visualize_freq = 1
        cfg.data.params.num_workers = 0
        for split in ("train", "val", "test"):
            cfg.data.params[split].source = str(source)
            cfg.data.params[split].target = str(target)
            cfg.data.params[split].batch_size = 1
        cfg.data.params.train.crop_size = 8
        params = cfg.model.params.layers.spectral.params
        params.channels = [4, 8, 16, 24]
        params.depths = [1, 1, 1]
        params.reconstruction_channels = [12, 8, 4]
        train_default(cfg)
        logs = tmp_path / cfg.experiment / "logs"
        assert (logs / "checkpoints" / "last.ckpt").is_file()
        assert (logs / "figures" / "reconst_0.png").is_file()
        metrics = (logs / "metrics.csv").read_text()
        assert "val_masked_mae" in metrics if phase == "pretrain" else "val_mrae" in metrics
        configs[phase] = cfg
    artifact = tmp_path / configs["pretrain"].experiment / "logs" / "checkpoints" / "encoder.pt"
    assert artifact.is_file()
    # Full restoration resume remains possible when the transfer artifact is gone.
    artifact.unlink()
    configs["restore"].resume = True
    configs["restore"].epochs = 3
    train_default(configs["restore"])
    ckpt = torch.load(
        tmp_path / configs["restore"].experiment / "logs" / "checkpoints" / "last.ckpt",
        weights_only=False,
    )
    assert ckpt["global_step"] == 3
