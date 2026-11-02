from pathlib import Path
import importlib

from hydra import compose, initialize_config_dir
import lightning as L
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch

from flows.core import DataSelector, ModelSelector, PipelineSelector
from flows.ml.datasets.volga2k.datamodule import paired_paths
from flows.ml.datasets.volga2k.dataset import Dataset
from flows.ml.pipelines.sprecher_rgb import SprecherRGBPipeline
from flows.cli.train import train_default


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def configuration(overrides=()):
    with initialize_config_dir(version_base="1.3.2",
                               config_dir=str(Path(__file__).resolve().parents[1] / ".configs")):
        return compose(config_name="sprecher_volga2k", overrides=list(overrides))


def images(root, size=(17, 19)):
    pixels = np.random.default_rng(42).integers(0, 256, (*size, 3), dtype=np.uint8)
    for split in ("train", "val", "test"):
        for side in ("source", "target"):
            folder = root / split / side
            folder.mkdir(parents=True)
            for name in ("b.png", "a.png"):
                Image.fromarray(pixels).save(folder / name)


def test_rgb_dataset_pairing_augmentation_eval_and_errors(tmp_path):
    images(tmp_path)
    a, b = paired_paths(tmp_path / "train/source", tmp_path / "train/target")
    assert Path(a[0]).name == "a.png"
    dataset = Dataset(a, b, crop_size=8, random_crop=True, flip_probability=0.5)
    x, y = dataset[0]
    assert x.shape == (3, 8, 8) and x.dtype == torch.float32
    torch.testing.assert_close(x, y)
    assert 0 <= x.min() <= x.max() <= 1
    evaluation = Dataset(a, b, crop_size=8, filename=True)
    torch.testing.assert_close(evaluation[0][0], evaluation[0][0])
    assert evaluation[0][2] == "a.png"
    assert Dataset(a, b)[0][0].shape == (3, 17, 19)
    with pytest.raises(ValueError, match="exceeds"):
        Dataset(a, b, crop_size=32)[0]
    (tmp_path / "train/target/a.png").unlink()
    with pytest.raises(ValueError, match="filenames"):
        paired_paths(tmp_path / "train/source", tmp_path / "train/target")


def test_rgb_dataset_rejects_bad_pairs(tmp_path):
    images(tmp_path)
    a, b = paired_paths(tmp_path / "train/source", tmp_path / "train/target")
    Image.new("RGB", (4, 4)).save(b[0])
    with pytest.raises(ValueError, match="shapes differ"):
        Dataset(a, b)[0]
    Path(b[0]).write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot read Volga2K image"):
        Dataset(a, b)[0]
    with pytest.raises(ValueError, match="nonzero"):
        Dataset([], [])
    with pytest.raises(ValueError, match="filenames"):
        Dataset(a, list(reversed(b)))


def test_volga2k_config_rgb_training_and_actual_test_cli(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    images(root)
    config = configuration([
        f"dataset_root={root.as_posix()}",
        f"save_dir={(tmp_path / 'experiments').as_posix()}",
        "accelerator=cpu", "devices=1", "epochs=1",
        "data.params.num_workers=0", "data.params.pin_memory=false",
        "data.params.train.crop_size=8", "data.params.val.crop_size=8",
        "model.params.layers.cmkan.params.mst_features=8",
        "model.params.layers.cmkan.params.mst_stages=1",
        "model.params.layers.cmkan.params.mst_head_channels=8",
    ])
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    dm = DataSelector.select(config.data)
    dm.setup("fit")
    assert dm.splits["val"]["source"] == f"{root.as_posix()}/val/source"
    model = ModelSelector.select(config.model)
    assert model.layers.cmkan.input_dim == model.layers.cmkan.output_dim == 3
    assert isinstance(model.layers.cmkan.input_projection, torch.nn.Identity)
    pipeline = PipelineSelector.select(model, config.pipeline)
    assert isinstance(pipeline, SprecherRGBPipeline)
    heads = [head.weight.detach().clone() for head in model.layers.cmkan.encoder.heads]
    assert all(torch.count_nonzero(head) == 0 for head in heads)
    batch = next(iter(dm.train_dataloader()))
    pipeline.restoration_losses(pipeline(batch[0]), batch[1])["loss"].backward()
    assert all(head.weight.grad is not None for head in model.layers.cmkan.encoder.heads)
    pipeline.zero_grad(set_to_none=True)
    trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, limit_train_batches=1,
                        limit_val_batches=1, num_sanity_val_steps=0)
    trainer.fit(pipeline, datamodule=dm)
    assert any(not torch.equal(old, head.weight)
               for old, head in zip(heads, model.layers.cmkan.encoder.heads))
    assert "val_de" in trainer.callback_metrics and "val_psnr" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["val_loss"])
    train_default(config)
    path = tmp_path / "experiments" / config.experiment / "logs/checkpoints/last.ckpt"
    assert path.is_file()
    # Exercise the actual test entrypoint with CPU-safe device settings.
    module = importlib.import_module("flows.cli.test")
    real_trainer = L.Trainer

    def test_trainer(**kwargs):
        kwargs.update(devices=1, limit_test_batches=1, enable_model_summary=False)
        return real_trainer(**kwargs)

    monkeypatch.setattr(module.L, "Trainer", test_trainer)
    module.main(config)


def test_missing_checkpoint_is_an_error(tmp_path):
    module = importlib.import_module("flows.cli.test")
    config = configuration([f"save_dir={tmp_path.as_posix()}"])
    with pytest.raises(FileNotFoundError, match="Test checkpoint"):
        module.main(config)


def test_rgb_quality_identity_and_huawei_alias():
    from flows.ml.datasets.huawei import DataModule as Old
    from flows.ml.datasets.volga2k import DataModule as New
    assert Old is New
    config = configuration()
    pipeline = PipelineSelector.select(ModelSelector.select(config.model), config.pipeline)
    image = torch.rand(1, 3, 12, 12)
    metrics = pipeline.quality_metrics(image, image)
    assert metrics["de"] == 0
    assert "sam" not in metrics
    assert pipeline.restoration_losses(image, image)["mae"] == 0
    assert pipeline.checkpoint_monitor == "val_loss"
