from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from flows.cli.train import train_default
from flows.core import ModelSelector, PipelineSelector
from flows.ml.models.ggpir import GGPIRCascade


def config():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / ".configs"), version_base="1.3",
    ):
        return compose(config_name="train_ggpir_cascade")


def test_direct_training_config_and_objective():
    torch.set_num_threads(1)
    cfg = config()
    pipeline = PipelineSelector.select(ModelSelector.select(cfg.model), cfg.pipeline)
    assert isinstance(pipeline.model.layers["cascade"], GGPIRCascade)
    assert pipeline.pretrain_epochs == 0
    assert pipeline.masked_val_batches == 0
    assert pipeline.checkpoint_monitor == "val_psnr"
    assert pipeline.checkpoint_mode == "max"
    assert cfg.data.params.train.batch_size == 4
    assert cfg.data.params.train.crop_size == 128
    x, y = torch.rand(1, 31, 16, 16), torch.rand(1, 31, 16, 16)
    expected = (x - y).abs().mean() + 0.15 * pipeline.ssim_loss(x, y)
    torch.testing.assert_close(pipeline.restoration_losses(x, y)["loss"], expected)
    pipeline._trainer = SimpleNamespace(max_epochs=20)
    optimizer = pipeline.configure_optimizers()
    assert optimizer["optimizer"].param_groups[0]["weight_decay"] == 0.05
    assert optimizer["lr_scheduler"].eta_min == 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cascade_cuda_gradients():
    model = GGPIRCascade().cuda().train()
    x = torch.rand(1, 31, 16, 16, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
        loss = (y.float() - x).square().mean()
    loss.backward()
    assert y.shape == x.shape and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_direct_training_entrypoint(tmp_path):
    torch.set_num_threads(1)
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    x = np.random.default_rng(17).random((16, 16, 31), dtype=np.float32)
    np.save(source / "sample.npy", x)
    np.save(target / "sample.npy", x * 0.8)
    cfg = config()
    cfg.accelerator = "cpu"
    cfg.devices = 1
    cfg.epochs = 1
    cfg.save_dir = str(tmp_path)
    cfg.resume = False
    cfg.data.params.num_workers = 0
    cfg.model.params.layers.cascade.params.n_ggp_per_scale = [1, 1, 1]
    for split in ("train", "val", "test"):
        cfg.data.params[split].source = str(source)
        cfg.data.params[split].target = str(target)
        cfg.data.params[split].batch_size = 1
    cfg.data.params.train.crop_size = 16
    train_default(cfg)
    path = tmp_path / cfg.experiment / "logs" / "checkpoints" / "last.ckpt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["global_step"] == 1
    assert any("'monitor': 'val_psnr', 'mode': 'max'" in key for key in ckpt["callbacks"])
    assert not (path.parent / "encoder.pt").exists()
