import os
import random
from pathlib import Path
from typing import Mapping, Optional

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Dataset


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def paired_paths(source: str, target: str):
    source_dir, target_dir = Path(source), Path(target)
    if not source_dir.is_dir() or not target_dir.is_dir():
        raise ValueError(f"Expected source/target directories: {source_dir}, {target_dir}.")
    sources = {p.name: str(p) for p in source_dir.iterdir()
               if p.is_file() and p.suffix.lower() == ".npy"}
    targets = {p.name: str(p) for p in target_dir.iterdir()
               if p.is_file() and p.suffix.lower() == ".npy"}
    if not sources or sources.keys() != targets.keys():
        raise ValueError(
            f"Source/target NPY filenames must match and be nonempty: "
            f"{source_dir}, {target_dir}; "
            f"missing targets={sorted(sources.keys() - targets.keys())[:5]}, "
            f"missing sources={sorted(targets.keys() - sources.keys())[:5]}."
        )
    names = sorted(sources)
    return [sources[n] for n in names], [targets[n] for n in names]


class DataModule(L.LightningDataModule):
    def __init__(
        self, train: Mapping, val: Optional[Mapping] = None,
        test: Optional[Mapping] = None, predict: Optional[Mapping] = None,
        num_workers: int = min(4, max(0, (os.cpu_count() or 1) - 2)),
        seed: int = 43, persistent_workers: bool = True,
        prefetch_factor: int = 2, pin_memory: bool = True,
        spectral_channels: int = 31, **kwargs,
    ) -> None:
        super().__init__()
        if num_workers < 0 or prefetch_factor < 1:
            raise ValueError("Require num_workers >= 0 and prefetch_factor >= 1.")
        val = val if val is not None else train
        test = test if test is not None else val
        predict = predict if predict is not None else test
        self.splits = {"train": train, "val": val, "test": test, "predict": predict}
        self.num_workers = num_workers
        self.seed = seed
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.spectral_channels = spectral_channels
        self.train_dataset = self.val_dataset = self.test_dataset = self.predict_dataset = None

    def _setup_split(self, split: str) -> None:
        if getattr(self, f"{split}_dataset") is not None:
            return
        config = self.splits[split]
        source, target = paired_paths(config["source"], config["target"])
        dataset = Dataset(
            source, target, filename=split == "predict",
            norm=config.get("norm", 1.0), crop_size=config.get("crop_size", 0),
            random_crop=split == "train",
            flip_probability=0.5 if split == "train" else 0.0,
            spectral_channels=self.spectral_channels,
        )
        setattr(self, f"{split}_dataset", dataset)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self._setup_split("train")
        if stage in ("fit", "validate", None):
            self._setup_split("val")
        if stage in ("test", None):
            self._setup_split("test")
        if stage in ("predict", None):
            self._setup_split("predict")

    def _loader(self, split: str) -> DataLoader:
        dataset = getattr(self, f"{split}_dataset")
        if dataset is None:
            raise RuntimeError(f"Call setup before requesting the {split} dataloader.")
        options = {}
        if self.num_workers > 0:
            options.update(
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                worker_init_fn=seed_worker,
            )
        return DataLoader(
            dataset, batch_size=self.splits[split]["batch_size"],
            shuffle=split == "train", num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(self.seed),
            **options,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader:
        return self._loader("test")

    def predict_dataloader(self) -> DataLoader:
        return self._loader("predict")
