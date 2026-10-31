from pathlib import Path
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, TensorDataset

from flows.core import ModelSelector, PipelineSelector
from flows.cli.train import train_default
from flows.ml.models.ggpir import (
    SpectralEncoderDecoder, ConvolutionalSpectralEncoderDecoder,
    GlobalCalibratedSpectralEncoderDecoder, IterativeSpectralEncoderDecoder,
)
from flows.ml.pipelines.spectral import SpectralExperimentPipeline


VARIANTS = [ConvolutionalSpectralEncoderDecoder, GlobalCalibratedSpectralEncoderDecoder,
            IterativeSpectralEncoderDecoder]


@pytest.fixture(autouse=True)
def small_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(threads)


def network(cls, head="pretrain", **kwargs):
    return cls(head=head, channels=8, stage_blocks=(1, 1, 1), **kwargs)


def config(name):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / ".configs"), version_base="1.3",
    ):
        return compose(config_name=name)


@pytest.mark.parametrize("cls", VARIANTS)
@pytest.mark.parametrize("head", ["pretrain", "finetune"])
@pytest.mark.parametrize("size", [(1, 1), (17, 19)])
def test_variant_shapes_gradients_and_hidden_input_isolation(cls, head, size):
    model = network(cls, head)
    x = torch.rand(2, 31, *size)
    mask = torch.zeros(2, 31, 1, 1, dtype=torch.bool)
    mask[:, ::2] = True
    output = model(x, mask)
    assert output.shape == x.shape and torch.isfinite(output).all()
    torch.testing.assert_close(output, model(x.masked_fill(mask, 999), mask))
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0
    if cls is IterativeSpectralEncoderDecoder:
        for module in (*model.encoder.stages, *model.encoder.refinement_heads):
            assert sum(p.grad.abs().sum() for p in module.parameters()) > 0


@pytest.mark.parametrize("cls", VARIANTS)
def test_checkpoint_recomputation_and_batch_independence(cls):
    model = network(cls).train()
    plain = network(cls, checkpoint_blocks=False).train()
    plain.load_state_dict(model.state_dict())
    x = torch.rand(2, 31, 13, 15)
    first, second = model(x), plain(x)
    torch.testing.assert_close(first, second)
    first.square().mean().backward()
    second.square().mean().backward()
    for p, q in zip(model.parameters(), plain.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x)[:1], model(x[:1]), atol=2e-6, rtol=2e-5)


def test_convolutional_variant_has_comparable_capacity():
    base = SpectralEncoderDecoder()
    conv = ConvolutionalSpectralEncoderDecoder()
    base_count = sum(p.numel() for p in base.parameters())
    conv_count = sum(p.numel() for p in conv.parameters())
    assert abs(conv_count / base_count - 1) < 0.05


def test_global_calibration_is_image_conditioned_and_identity_when_zeroed():
    model = network(GlobalCalibratedSpectralEncoderDecoder).eval()
    base = network(SpectralEncoderDecoder).eval()
    base.load_state_dict({k: v for k, v in model.state_dict().items()
                          if not k.startswith("encoder.calibration.")})
    x = torch.rand(1, 31, 17, 19)
    with torch.no_grad():
        assert not torch.allclose(model(x), base(x))
        model.encoder.calibration[-1].weight.zero_()
        model.encoder.calibration[-1].bias.zero_()
        torch.testing.assert_close(model(x), base(x))


@pytest.mark.parametrize("iterations", [0, 1, 1.5, True])
def test_iterative_rejects_invalid_stage_count(iterations):
    with pytest.raises(ValueError, match="iterations"):
        network(IterativeSpectralEncoderDecoder, iterations=iterations)


def test_iterative_stages_receive_cumulative_spectral_corrections():
    model = network(IterativeSpectralEncoderDecoder, "finetune", iterations=3).eval()
    x = torch.rand(1, 31, 9, 11)
    inputs = []
    handles = [stage.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
               for stage in model.encoder.stages[1:]]
    try:
        with torch.no_grad():
            for head, correction in zip(model.encoder.refinement_heads, (0.1, 0.2)):
                head.weight.zero_()
                head.bias.fill_(correction)
            model.decoder.output.weight.zero_()
            model.decoder.output.bias.zero_()
            result = model(x)
        torch.testing.assert_close(inputs[0], x + 0.1)
        torch.testing.assert_close(inputs[1], x + 0.3)
        torch.testing.assert_close(result, x + 0.3)
        assert len({next(stage.parameters()).data_ptr()
                    for stage in model.encoder.stages}) == 3
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("cls", VARIANTS)
@pytest.mark.parametrize("head", ["pretrain", "finetune"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_default_variant_cuda_forward_backward(cls, head, dtype):
    model = cls(head=head).cuda()
    x = torch.rand(8, 31, 196, 196, device="cuda")
    mask = torch.zeros(8, 31, 1, 1, dtype=torch.bool, device="cuda")
    mask[:, ::2] = True
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        output = model(x, mask)
        loss = (output.float() - x).square().mean()
    loss.backward()
    assert output.shape == x.shape and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    del model, output, loss, x, mask
    torch.cuda.empty_cache()


@pytest.mark.parametrize("cls", VARIANTS)
def test_pretraining_transfer_freeze_resume_and_cross_variant_rejection(cls, tmp_path):
    x = torch.rand(2, 31, 8, 8)
    loader = DataLoader(TensorDataset(x, x * 0.8), batch_size=1)

    def trainer(root):
        return L.Trainer(default_root_dir=root, accelerator="cpu", devices=1,
                         max_epochs=1, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0)

    pretrain = SpectralExperimentPipeline(network(cls), phase="pretrain")
    trainer(tmp_path / "pretrain").fit(pretrain, loader, loader)
    artifact = tmp_path / "pretrain" / "checkpoints" / "encoder.pt"
    finetune = SpectralExperimentPipeline(network(cls, "finetune"), phase="finetune",
                                         encoder_checkpoint=str(artifact),
                                         freeze_encoder_steps=1)
    run = trainer(tmp_path / "finetune")
    run.fit(finetune, loader, loader)
    p = next(finetune.network.encoder.parameters())
    assert run.optimizers[0].state[p]["step"] == 1
    ckpt = tmp_path / "full.ckpt"
    run.save_checkpoint(ckpt)
    resumed = SpectralExperimentPipeline(network(cls, "finetune"), phase="finetune",
                                        freeze_encoder_steps=1)
    resumed.on_load_checkpoint(torch.load(ckpt, weights_only=False))
    assert resumed._resuming
    wrong = SpectralExperimentPipeline(network(SpectralEncoderDecoder, "finetune"),
                                       phase="finetune", freeze_encoder_steps=1)
    with pytest.raises(ValueError, match="Incompatible"):
        wrong.load_encoder(str(artifact))
    with pytest.raises(ValueError, match="matching spectral encoder and decoder"):
        wrong.on_load_checkpoint(torch.load(ckpt, weights_only=False))


@pytest.mark.parametrize("suffix,cls", [
    ("conv", ConvolutionalSpectralEncoderDecoder),
    ("global", GlobalCalibratedSpectralEncoderDecoder),
    ("iterative", IterativeSpectralEncoderDecoder),
])
def test_paired_variant_configs_and_artifact_paths(suffix, cls):
    pretrain, finetune = config(f"pretrain_{suffix}"), config(f"finetune_{suffix}")
    for cfg in (pretrain, finetune):
        module = PipelineSelector.select(ModelSelector.select(cfg.model), cfg.pipeline)
        assert isinstance(module.network, cls)
    assert finetune.pipeline.params.encoder_checkpoint == (
        f"{pretrain.save_dir}/{pretrain.experiment}/logs/checkpoints/encoder.pt")
    assert pretrain.experiment != config("pretrain").experiment


@pytest.mark.parametrize("name", ["finetune_scratch", "finetune_fast_encoder", "finetune_psnr"])
def test_training_ablations_isolate_changes(name):
    cfg, baseline = config(name), config("finetune")
    assert cfg.model == baseline.model and cfg.data == baseline.data
    assert cfg.experiment != baseline.experiment
    if name == "finetune_scratch":
        assert cfg.pipeline.params.encoder_checkpoint is None
        assert cfg.pipeline.params.allow_random_encoder
        assert cfg.pipeline.params.freeze_encoder_steps == baseline.pipeline.params.freeze_encoder_steps
    else:
        assert cfg.pipeline.params.encoder_checkpoint == baseline.pipeline.params.encoder_checkpoint
    if name == "finetune_fast_encoder":
        assert cfg.pipeline.params.encoder_lr == cfg.pipeline.params.lr
    if name == "finetune_psnr":
        module = PipelineSelector.select(ModelSelector.select(cfg.model), cfg.pipeline)
        pred, target = torch.rand(1, 31, 8, 8, requires_grad=True), torch.rand(1, 31, 8, 8)
        loss = module.restoration_losses(pred, target)["loss"]
        torch.testing.assert_close(loss, (pred - target).square().mean())
        loss.backward()
        torch.testing.assert_close(pred.grad, 2 * (pred.detach() - target) / pred.numel())
        assert module.checkpoint_monitor == "val_psnr" and module.checkpoint_mode == "max"


def test_objective_ablation_cannot_silently_resume_another_objective():
    base = SpectralExperimentPipeline(network(SpectralEncoderDecoder), phase="pretrain")
    base._trainer = SimpleNamespace(callback_metrics={})
    ckpt = {}
    base.on_save_checkpoint(ckpt)
    changed = SpectralExperimentPipeline(network(SpectralEncoderDecoder), phase="pretrain",
                                        mse_weight=1)
    with pytest.raises(ValueError, match="objective ablation"):
        changed.on_load_checkpoint(ckpt)


def test_psnr_entrypoint_saves_maximizing_checkpoint(tmp_path):
    import numpy as np
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    cube = np.random.default_rng(1).random((8, 8, 31), dtype=np.float32)
    np.save(source / "sample.npy", cube)
    np.save(target / "sample.npy", cube * 0.8)
    cfg = config("finetune_psnr")
    cfg.save_dir = str(tmp_path)
    cfg.accelerator = "cpu"
    cfg.devices = 1
    cfg.epochs = 2
    cfg.resume = False
    cfg.data.params.num_workers = 0
    cfg.model.params.layers.spectral.params.channels = 8
    cfg.model.params.layers.spectral.params.stage_blocks = [1, 1, 1]
    cfg.pipeline.params.encoder_checkpoint = None
    cfg.pipeline.params.allow_random_encoder = True
    for split in ("train", "val", "test"):
        cfg.data.params[split].source = str(source)
        cfg.data.params[split].target = str(target)
        cfg.data.params[split].batch_size = 1
    cfg.data.params.train.crop_size = 8
    train_default(cfg)
    ckpt = torch.load(tmp_path / cfg.experiment / "logs" / "checkpoints" / "last.ckpt",
                      weights_only=False)
    callbacks = [v for k, v in ckpt["callbacks"].items() if "ModelCheckpoint" in k]
    assert len(callbacks) == 1
    assert callbacks[0]["best_model_score"] is not None
    assert "'mode': 'max'" in next(k for k in ckpt["callbacks"] if "ModelCheckpoint" in k)
    import csv
    with (tmp_path / cfg.experiment / "logs" / "metrics.csv").open() as stream:
        scores = [float(row["val_psnr"]) for row in csv.DictReader(stream) if row["val_psnr"]]
    assert float(callbacks[0]["best_model_score"]) == pytest.approx(max(scores))
