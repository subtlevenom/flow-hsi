from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from flows.core.selector import DataSelector
from flows.ml.callbacks.generate import GenerateCallback
from flows.ml.datasets.ntire.datamodule import DataModule, paired_paths
from flows.ml.datasets.ntire.dataset import Dataset
from flows.ml.pipelines.usgs_unet import USGSUNetPipeline


def cubes(tmp_path, shape=(20, 24, 31), dtype=np.float32):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    cube = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(dtype)
    np.save(source / "sample.npy", cube)
    np.save(target / "sample.npy", cube)
    return source, target, cube


def config(source, target, crop_size=8):
    return {"source": str(source), "target": str(target), "batch_size": 1,
            "crop_size": crop_size}


def test_maps_are_read_only_and_only_cropped_chw_patches_are_copied(tmp_path, monkeypatch):
    source, target, cube = cubes(tmp_path)
    original_load, original_array = np.load, np.array
    loads, copies = [], []

    def load(*args, **kwargs):
        array = original_load(*args, **kwargs)
        loads.append((kwargs, isinstance(array, np.memmap), array.flags.writeable))
        return array

    def array(value, *args, **kwargs):
        copies.append(value.shape)
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    monkeypatch.setattr(np, "array", array)
    dataset = Dataset(*paired_paths(str(source), str(target)), crop_size=8, norm=2)
    x, y = dataset[0]
    assert len(loads) == 2
    assert all(options == {"mmap_mode": "r", "allow_pickle": False}
               and is_map and not writable for options, is_map, writable in loads)
    assert copies == [(31, 8, 8), (31, 8, 8)]
    expected = torch.from_numpy(cube[6:14, 8:16].transpose(2, 0, 1).copy())
    torch.testing.assert_close(x, expected / 2)
    torch.testing.assert_close(y, expected)
    assert x.is_contiguous() and y.is_contiguous()
    x[0, 0, 0] = -1
    np.testing.assert_array_equal(original_load(source / "sample.npy"), cube)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8, np.uint16])
def test_preserves_values_dtype_scaling_alignment_and_filename(tmp_path, dtype):
    source, target, cube = cubes(tmp_path, dtype=dtype)
    dataset = Dataset(*paired_paths(str(source), str(target)),
                      crop_size=8, filename=True)
    x, y, name = dataset[0]
    assert name == "sample.npy"
    assert x.dtype == y.dtype == torch.float32
    assert x.shape == y.shape == (31, 8, 8)
    torch.testing.assert_close(x, y)
    legacy_transform = Compose([ToImage(), ToDtype(dtype=torch.float32, scale=True)])
    expected = legacy_transform(cube)[..., 6:14, 8:16]
    torch.testing.assert_close(x, expected)


def test_training_randomness_and_evaluation_center_crop(tmp_path):
    source, target, cube = cubes(tmp_path)
    dm = DataModule(config(source, target), num_workers=0)
    dm.setup("fit")
    outputs = [dm.train_dataset[0][0].clone() for _ in range(8)]
    assert any(not torch.equal(outputs[0], x) for x in outputs[1:])
    for _ in range(3):
        x, y = dm.val_dataset[0]
        expected = torch.from_numpy(cube[6:14, 8:16].transpose(2, 0, 1).copy())
        torch.testing.assert_close(x, expected)
        torch.testing.assert_close(y, expected)
    dm.setup("test")
    dm.setup("predict")
    torch.testing.assert_close(dm.test_dataset[0][0], expected)
    torch.testing.assert_close(dm.predict_dataset[0][0], expected)


def test_full_image_retains_existing_multiple_of_four_convention(tmp_path):
    source, target, cube = cubes(tmp_path, shape=(19, 23, 31))
    dataset = Dataset(*paired_paths(str(source), str(target)))
    x, y = dataset[0]
    expected = torch.from_numpy(cube[:16, :20].transpose(2, 0, 1).copy())
    torch.testing.assert_close(x, expected)
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("problem", ["different_names", "missing", "empty", "missing_dir"])
def test_invalid_pair_directories_are_rejected(tmp_path, problem):
    source, target, _ = cubes(tmp_path)
    if problem == "different_names":
        (target / "sample.npy").rename(target / "wrong.npy")
    elif problem == "missing":
        (target / "sample.npy").unlink()
    elif problem == "empty":
        (target / "sample.npy").unlink()
        (source / "sample.npy").unlink()
    else:
        source = tmp_path / "missing"
    with pytest.raises(ValueError, match="directories|filenames"):
        paired_paths(str(source), str(target))


@pytest.mark.parametrize("shape", [(20, 23, 31), (20, 24, 30), (20, 24), (1, 24, 31)])
def test_mismatched_or_invalid_cube_shapes_fail_with_paths(tmp_path, shape):
    source, target, _ = cubes(tmp_path)
    np.save(target / "sample.npy", np.zeros(shape, dtype=np.float32))
    dataset = Dataset(*paired_paths(str(source), str(target)), crop_size=8)
    with pytest.raises(ValueError, match="matching HWC.*sample.npy"):
        dataset[0]


def test_corrupt_cube_fails_with_path_instead_of_none(tmp_path):
    source, target, _ = cubes(tmp_path)
    (source / "sample.npy").write_bytes(b"not an npy file")
    dataset = Dataset(*paired_paths(str(source), str(target)))
    with pytest.raises(ValueError, match="Cannot memory-map NTIRE cube.*sample.npy"):
        dataset[0]


@pytest.mark.parametrize("kwargs", [
    {"norm": 0}, {"norm": float("nan")}, {"crop_size": -1},
    {"crop_size": 2}, {"flip_probability": 1.1},
])
def test_bad_dataset_settings_are_rejected(tmp_path, kwargs):
    source, target, _ = cubes(tmp_path)
    with pytest.raises(ValueError):
        Dataset(*paired_paths(str(source), str(target)), **kwargs)


def test_crop_larger_than_image_is_rejected(tmp_path):
    source, target, _ = cubes(tmp_path)
    dataset = Dataset(*paired_paths(str(source), str(target)), crop_size=32)
    with pytest.raises(ValueError, match="exceeds cube size"):
        dataset[0]


def test_split_setup_is_lazy_and_zero_workers_omits_worker_only_options(tmp_path):
    source, target, _ = cubes(tmp_path)
    missing = config(tmp_path / "missing", tmp_path / "missing")
    dm = DataModule(config(source, target), test=missing, num_workers=0,
                    persistent_workers=True, prefetch_factor=1)
    dm.setup("fit")
    before = dm.train_dataset
    dm.setup("fit")
    assert dm.train_dataset is before
    for loader in (dm.train_dataloader(), dm.val_dataloader()):
        assert loader.num_workers == 0
        assert not loader.persistent_workers
        assert loader.prefetch_factor is None
        assert len(list(loader)) == 1
    with pytest.raises(ValueError, match="directories"):
        dm.setup("test")


def test_persistent_workers_reuse_processes_and_produce_seeded_aligned_crops(tmp_path):
    source, target, _ = cubes(tmp_path)

    def make_loader():
        dm = DataModule(config(source, target), num_workers=2, prefetch_factor=1,
                        persistent_workers=True, pin_memory=False, seed=19)
        dm.setup("fit")
        return dm.train_dataloader()

    first = make_loader()
    second = make_loader()
    try:
        x, y = next(iter(first))
        torch.testing.assert_close(x, y)
        worker_pids = [worker.pid for worker in first._iterator._workers]
        reproducible, _ = next(iter(second))
        torch.testing.assert_close(x, reproducible)
        next(iter(first))
        assert [worker.pid for worker in first._iterator._workers] == worker_pids
        assert all(worker.is_alive() for worker in first._iterator._workers)
        assert first.prefetch_factor == 1
    finally:
        for loader in (first, second):
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()


def test_preview_reuses_output_without_forward_loader_iteration_or_mode_change(tmp_path):
    callback = GenerateCallback(every_n_epochs=1, max_images=1, max_size=8)
    trainer = SimpleNamespace(
        is_global_zero=True, sanity_checking=False, current_epoch=0,
        log_dir=str(tmp_path), default_root_dir=str(tmp_path),
    )
    model = Mock(side_effect=AssertionError("Preview must reuse the existing prediction"))
    src, target = torch.rand(2, 31, 20, 24), torch.rand(2, 31, 20, 24)
    callback.on_validation_batch_end(trainer, model, {"image": src}, (src, target), 0)
    assert (tmp_path / "figures" / "reconst_0.png").is_file()
    model.assert_not_called()
    from PIL import Image
    with Image.open(tmp_path / "figures" / "reconst_0.png") as image:
        assert image.width < 40 and image.height < 20
    callback.on_test_batch_end(trainer, model, {"image": src}, (src, target), 0)
    assert (tmp_path / "figures" / "test_0.png").is_file()


@pytest.mark.parametrize("change", [
    {"is_global_zero": False}, {"sanity_checking": True}, {"current_epoch": 1},
])
def test_preview_skips_unneeded_work(tmp_path, change):
    trainer = SimpleNamespace(
        is_global_zero=True, sanity_checking=False, current_epoch=0,
        log_dir=str(tmp_path), default_root_dir=str(tmp_path),
    )
    for key, value in change.items():
        setattr(trainer, key, value)
    callback = GenerateCallback(every_n_epochs=5)
    callback.on_validation_batch_end(trainer, Mock(), {}, None, 0)
    assert not (tmp_path / "figures").exists()


def test_masked_validation_is_bounded_and_primary_loss_is_paired(monkeypatch):
    pipeline = USGSUNetPipeline(torch.nn.Identity(), masked_val_batches=1)
    src, target = torch.rand(1, 31, 8, 8), torch.rand(1, 31, 8, 8)
    forward = Mock(side_effect=lambda x: x)
    monkeypatch.setattr(pipeline, "forward", forward)
    monkeypatch.setattr(pipeline, "log", lambda *args, **kwargs: None)
    first = pipeline.validation_step((src, target), 0)
    assert forward.call_count == 2
    torch.testing.assert_close(first["loss"], pipeline.restoration_losses(src, target)["loss"])
    assert "image" in first
    second = pipeline.validation_step((src, target), 1)
    assert forward.call_count == 3
    assert "image" not in second


def test_preview_fallback_is_bounded_without_changing_model_mode(tmp_path):
    callback = GenerateCallback(every_n_epochs=1, max_images=1, max_size=8)
    trainer = SimpleNamespace(
        is_global_zero=True, sanity_checking=False, current_epoch=0,
        log_dir=str(tmp_path), default_root_dir=str(tmp_path),
    )
    model = torch.nn.Identity().eval()
    sizes = []
    model.register_forward_hook(lambda module, args, result: sizes.append(args[0].shape))
    src = torch.rand(2, 31, 20, 24)
    callback.on_validation_batch_end(trainer, model, {"loss": torch.tensor(0)}, (src, src), 0)
    assert sizes == [(1, 31, 8, 8)]
    assert not model.training
    assert (tmp_path / "figures" / "reconst_0.png").is_file()


def test_disabling_masked_validation_avoids_the_extra_model_pass(monkeypatch):
    pipeline = USGSUNetPipeline(torch.nn.Identity(), masked_val_batches=0)
    forward = Mock(side_effect=lambda x: x)
    monkeypatch.setattr(pipeline, "forward", forward)
    monkeypatch.setattr(pipeline, "log", lambda *args, **kwargs: None)
    src = torch.rand(1, 31, 8, 8)
    pipeline.validation_step((src, src), 0)
    assert forward.call_count == 1


def test_preview_frequency_aligns_with_validation_every_five_epochs(tmp_path):
    callback = GenerateCallback(every_n_epochs=5)
    trainer = SimpleNamespace(
        is_global_zero=True, sanity_checking=False, current_epoch=4,
        log_dir=str(tmp_path), default_root_dir=str(tmp_path),
    )
    src = torch.rand(1, 31, 8, 8)
    callback.on_validation_batch_end(trainer, Mock(), {"image": src}, (src, src), 0)
    assert (tmp_path / "figures" / "reconst_4.png").is_file()
