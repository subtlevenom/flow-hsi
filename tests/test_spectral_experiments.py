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
    ChannelLayerNorm, LocalBlock, PixelQueryBlock, ResidualConvBlock, SpectralEncoderDecoder,
)
from flows.ml.pipelines.spectral import SpectralExperimentPipeline


@pytest.fixture(autouse=True)
def cpu_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(threads)


def network(head="pretrain", **options):
    params = dict(
        channels=8,
        branch_dilations=((1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)),
        branch_channels=(8, 12, 16, 20), heads=2, pixel_chunk_size=128,
    )
    params.update(options)
    return SpectralEncoderDecoder(head=head, **params)


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


@pytest.mark.parametrize("head", ["pretrain", "finetune"])
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


def test_full_resolution_branches_and_pixel_query_fusion():
    model = network()
    x = torch.rand(1, 31, 17, 19)
    observed = []
    handles = [
        path[-1].register_forward_hook(lambda module, inputs, output: observed.append(output.shape))
        for path in [*model.encoder.branches, model.encoder.detail]
    ]
    features = model.encoder(x)
    for handle in handles:
        handle.remove()
    assert features.shape == (1, 8, 17, 19)
    assert observed == [
        (1, 8, 17, 19), (1, 12, 17, 19), (1, 16, 17, 19),
        (1, 20, 17, 19), (1, 8, 17, 19),
    ]
    assert [len(path) for path in model.encoder.branches] == [1, 2, 3, 4]
    assert [[block.body[0].dilation[0] for block in path]
            for path in model.encoder.branches] == [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]
    assert all(isinstance(path[0], ResidualConvBlock) for path in model.encoder.branches)
    assert model.encoder.branch_embedding.shape == (6, 8)
    assert len(model.encoder.fusion) == 2
    assert all(isinstance(block, PixelQueryBlock) for block in model.encoder.fusion)
    assert all(isinstance(block, LocalBlock) for block in model.decoder.blocks)
    assert network("finetune").decoder.output.weight.count_nonzero() > 0


def test_pixel_query_attention_does_not_mix_pixels_and_uses_context():
    block = PixelQueryBlock(8, 2).eval()
    query = torch.rand(3, 8)
    tokens = torch.rand(3, 6, 8)
    expected = block(query, tokens)
    changed = tokens.clone()
    changed[1, 2] += torch.arange(8).float()
    actual = block(query, changed)
    torch.testing.assert_close(actual[[0, 2]], expected[[0, 2]])
    assert not torch.allclose(actual[1], expected[1])
    assert not torch.allclose(block(query + torch.arange(8), tokens), expected)


@pytest.mark.parametrize("head", ["pretrain", "finetune"])
def test_common_pixels_are_invariant_for_unaligned_crops_and_other_images(head):
    model = network(head).eval()
    x = torch.rand(1, 31, 192, 196)
    with torch.no_grad():
        full = model(x)
        patch = model(x[..., 7:183, 7:187])
        batched = model(torch.cat([x, torch.rand_like(x)]))
    torch.testing.assert_close(patch[..., 48:-48, 48:-48],
                               full[..., 55:135, 55:139], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(batched[:1], full, atol=2e-6, rtol=2e-5)


def test_chunking_and_checkpointing_preserve_predictions_and_gradients():
    model = network().train()
    plain = network(checkpoint_blocks=False).train()
    plain.load_state_dict(model.state_dict())
    plain.encoder.pixel_chunk_size = 10000
    x = torch.rand(2, 31, 13, 15)
    expected = plain(x)
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, p), (other_name, other) in zip(model.named_parameters(), plain.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p.grad, other.grad, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("head", ["pretrain", "finetune"])
def test_training_crop_196_matches_full_256_common_center(head):
    model = network(head).eval()
    x = torch.rand(1, 31, 256, 256)
    with torch.no_grad():
        full = model(x)
        patch = model(x[..., 27:223, 27:223])
    torch.testing.assert_close(patch[..., 64:132, 64:132],
                               full[..., 91:159, 91:159], atol=2e-6, rtol=2e-5)


def test_output_uses_neighbor_spectra_but_has_bounded_spatial_receptive_field():
    model = network().eval()
    x = torch.rand(1, 31, 128, 128, requires_grad=True)
    model(x)[0, 0, 64, 64].backward()
    assert x.grad[0, 1:, 64, 64].abs().sum() > 0
    assert x.grad[..., 64, 65].abs().sum() > 0
    assert x.grad[..., :16, :].count_nonzero() == 0
    assert x.grad[..., 112:, :].count_nonzero() == 0


def test_finetune_head_is_identity_residual_and_pretrain_has_no_input_bypass():
    x = torch.rand(1, 31, 9, 11)
    for head in ("pretrain", "finetune"):
        model = network(head)
        with torch.no_grad():
            model.decoder.output.weight.zero_()
            model.decoder.output.bias.zero_()
        torch.testing.assert_close(model(x), x if head == "finetune" else torch.zeros_like(x))


@pytest.mark.parametrize("options", [
    {"head_depth": 0}, {"head_depth": 1.5}, {"detail_depth": 0},
    {"fusion_depth": 0}, {"branch_dilations": []},
    {"branch_dilations": [[1], [0], [1], [1]]},
    {"branch_dilations": [[1], [1], [1]]},
    {"branch_channels": [1]}, {"branch_channels": [8, 0, 8, 8]},
    {"branch_channels": [8, 8, 8]},
    {"checkpoint_blocks": 1},
    {"channels": 0}, {"channels": 7}, {"heads": 0},
    {"pixel_chunk_size": 0}, {"spectral_channels": 1},
])
def test_model_options_reject_invalid_values(options):
    with pytest.raises(ValueError):
        network("finetune", **options)


def test_resume_checks_decoder_and_rejects_legacy_architecture():
    module = pipeline("finetune")
    module._trainer = SimpleNamespace(callback_metrics={})
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    different = SpectralExperimentPipeline(
        network("finetune", head_depth=1), phase="finetune",
    )
    with pytest.raises(ValueError, match="decoder architecture"):
        different.on_load_checkpoint(checkpoint)
    del checkpoint["spectral_experiment"]["model_architecture"]
    with pytest.raises(ValueError, match="Legacy checkpoints"):
        module.on_load_checkpoint(checkpoint)


def test_previous_unet_encoder_artifacts_are_rejected(tmp_path):
    module = pipeline("finetune")
    path = tmp_path / "old.pt"
    torch.save({
        "format": "flow-hsi.spectral-encoder.v1",
        "architecture": {"version": 2, "calibration": {"type": "residual_pointwise_mlp"}},
        "encoder_state_dict": {},
    }, path)
    with pytest.raises(ValueError, match="requires new pretraining"):
        module.load_encoder(str(path))


@pytest.mark.parametrize("head", ["pretrain", "finetune"])
def test_model_masks_every_path_without_hidden_target_bypass(head):
    model = network(head)
    x = torch.rand(1, 31, 8, 8)
    mask = torch.zeros(1, 31, 1, 1, dtype=torch.bool)
    mask[:, ::2] = True
    other = x.clone()
    other[:, ::2] = 999
    torch.testing.assert_close(
        model(x, mask), model(other, mask),
    )
    assert model.encoder.architecture()["version"] == 6


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


@pytest.mark.parametrize("head", ["pretrain", "finetune"])
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
    finetune = pipeline("finetune", encoder_checkpoint=str(path))
    before = {key: value.clone() for key, value in finetune.network.decoder.state_dict().items()}
    finetune.load_encoder(str(path))
    for key, value in finetune.network.encoder.state_dict().items():
        torch.testing.assert_close(value, artifact["encoder_state_dict"][key])
    for key, value in finetune.network.decoder.state_dict().items():
        torch.testing.assert_close(value, before[key])
    artifact["architecture"]["channels"] = 999
    torch.save(artifact, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="Incompatible"):
        finetune.load_encoder(str(tmp_path / "wrong.pt"))
    with pytest.raises(FileNotFoundError):
        finetune.load_encoder(str(tmp_path / "missing.pt"))


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
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


def test_finetune_sharpness_loss_matches_spatial_edges_and_is_weighted():
    module = pipeline("finetune", sharpness_weight=1.0, ssim_weight=0)
    target = torch.zeros(1, 31, 8, 8)
    target[..., 2:6, 2:6] = 1
    blurred = torch.nn.functional.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sharp_terms = module.restoration_losses(target, target)
    blurred_terms = module.restoration_losses(blurred.requires_grad_(), target)
    torch.testing.assert_close(sharp_terms["sharpness_loss"], torch.tensor(0.0))
    assert blurred_terms["sharpness_loss"] > 0
    module.sharpness_weight = 0
    base = module.restoration_losses(blurred.detach(), target)["loss"]
    module.sharpness_weight = 1.0
    torch.testing.assert_close(
        blurred_terms["loss"], base + blurred_terms["sharpness_loss"],
    )
    blurred_terms["sharpness_loss"].backward()
    assert torch.isfinite(blurred.grad).all()
    assert blurred.grad.abs().sum() > 0


@pytest.mark.parametrize("size", [(1, 1), (1, 4), (4, 1)])
def test_sharpness_loss_is_finite_for_degenerate_spatial_sizes(size):
    pred = torch.rand(1, 31, *size, requires_grad=True)
    target = torch.rand_like(pred)
    loss = pipeline("finetune").sharpness_loss(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


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
        pipeline("finetune").on_load_checkpoint(checkpoint)
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
    module = pipeline("finetune", encoder_checkpoint=str(path), freeze_encoder_steps=1)
    record = FreezeRecord()
    run = trainer(tmp_path / "finetune", callbacks=[record], enable_checkpointing=False)
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
    module = pipeline("finetune", allow_random_encoder=True, freeze_encoder_steps=500)
    record = FreezeRecord()
    run = trainer(tmp_path, callbacks=[record], enable_checkpointing=False)
    run.fit(module, loader(), loader())
    assert [entry["frozen"] for entry in record.records] == [True, True, False, False]


def test_query_fusion_follows_encoder_freeze_and_unfreeze(tmp_path):
    module = pipeline("finetune", allow_random_encoder=True, freeze_encoder_steps=1)
    head = module.network.encoder.fusion[0]
    before = head.query.weight.detach().clone()

    class RecordFusion(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step == 1:
                torch.testing.assert_close(head.query.weight, before)
                assert head.query.weight.grad is None

    run = trainer(tmp_path, callbacks=[RecordFusion()], enable_checkpointing=False)
    run.fit(module, loader(), loader())
    assert not torch.equal(head.query.weight, before)
    optimizer = run.optimizers[0]
    encoder_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    assert all(id(p) in encoder_ids for p in head.parameters())
    assert optimizer.state[head.query.weight]["step"] == 3


def test_freeze_counts_optimizer_updates_with_accumulation(tmp_path):
    module = pipeline("finetune", allow_random_encoder=True, freeze_encoder_steps=1)
    record = FreezeRecord()
    run = trainer(tmp_path, callbacks=[record], enable_checkpointing=False,
                  accumulate_grad_batches=2)
    run.fit(module, loader(), loader())
    assert [entry["frozen"] for entry in record.records] == [True, True, False, False]
    assert run.global_step == 2


def test_resume_restores_frozen_phase_and_does_not_reload_encoder_artifact(tmp_path):
    _, path = export_encoder(tmp_path / "pretrain")
    artifact = str(path)
    module = pipeline("finetune", encoder_checkpoint=artifact, freeze_encoder_steps=2)
    checkpoint = ModelCheckpoint(monitor="val_mrae", save_last=True)
    record = FreezeRecord()

    class SaveMidEpoch(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            trainer.save_checkpoint(tmp_path / "mid_epoch.ckpt")

    first = trainer(tmp_path / "finetune", callbacks=[
        checkpoint, record, SaveMidEpoch(),
    ], max_steps=1)
    first.fit(module, loader(), loader())
    assert record.records[0]["frozen"]
    path.unlink()
    resumed = pipeline("finetune", encoder_checkpoint=artifact, freeze_encoder_steps=2)
    resumed_record = FreezeRecord()
    second = trainer(tmp_path / "finetune", callbacks=[
        resumed_record, ModelCheckpoint(monitor="val_mrae", save_last=True),
    ])
    second.fit(resumed, loader(), loader(), ckpt_path=tmp_path / "mid_epoch.ckpt")
    assert resumed._resuming
    assert resumed_record.records[0]["frozen"]
    assert any(not item["frozen"] for item in resumed_record.records)
    assert second.global_step >= 3


def test_restore_without_encoder_does_not_silently_train_random(tmp_path):
    module = pipeline("finetune")
    with pytest.raises(ValueError, match="requires encoder_checkpoint"):
        trainer(tmp_path, enable_checkpointing=False).fit(module, loader())


@pytest.mark.parametrize("phase,epochs", [
    ("pretrain", 31), ("finetune", 71),
])
def test_experiment_config_and_factory(phase, epochs):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / ".configs"), version_base="1.3",
    ):
        cfg = compose(config_name=phase)
    assert cfg.epochs == epochs
    assert cfg.pipeline.params.ssim_weight == 0.15
    assert cfg.pipeline.params.get("sharpness_weight", 0.0) == (
        1.0 if phase == "finetune" else 0.0
    )
    module = PipelineSelector.select(ModelSelector.select(cfg.model), cfg.pipeline)
    assert isinstance(module, SpectralExperimentPipeline)
    assert module.checkpoint_monitor == ("val_masked_mae" if phase == "pretrain" else "val_mrae")
    assert len(module.network.encoder.branches) == 4
    assert module.network.encoder.channels == 64
    assert module.network.encoder.branch_channels == (48, 96, 128, 160)
    assert module.network.encoder.architecture()["version"] == 6
    x = torch.rand(1, 31, 17, 19)
    assert module(x).shape == x.shape
    assert module.model(src=x)["res"].shape == x.shape
    if phase != "pretrain":
        assert len(module.network.decoder.blocks) == 2
        assert cfg.experiment == "spectral.finetune.gazetr.fullrf.wide.ntire"


def test_two_experiments_through_real_ntire_entrypoint(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    cube = np.random.default_rng(1).random((16, 16, 31), dtype=np.float32)
    np.save(source / "sample.npy", cube)
    np.save(target / "sample.npy", cube * 0.8)
    configs = {}
    for phase in ("pretrain", "finetune"):
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
        params.channels = 8
        params.branch_dilations = [[1], [1], [1], [1]]
        params.branch_channels = [8, 12, 16, 20]
        params.heads = 2
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
    configs["finetune"].resume = True
    configs["finetune"].epochs = 3
    train_default(configs["finetune"])
    ckpt = torch.load(
        tmp_path / configs["finetune"].experiment / "logs" / "checkpoints" / "last.ckpt",
        weights_only=False,
    )
    assert ckpt["global_step"] == 3
