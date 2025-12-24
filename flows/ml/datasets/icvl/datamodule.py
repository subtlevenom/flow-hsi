import os
import random
import torch
import lightning as L
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    CenterCrop,
    RandomCrop,
)
from torch.utils.data import DataLoader
from typing import Tuple
from .dataset import Dataset
from flows.core import Logger
from flows.ml.transforms.pair_trransform import PairTransform

CROP = 128
IMG_EXTS: Tuple[str] = (".npy")

class DataModule(L.LightningDataModule):
    def __init__(
            self,
            train: dict,
            val: dict = None,
            test: dict = None,
            num_workers: int = min(12, os.cpu_count() - 2),
            seed: int = 43,
    ) -> None:
        super().__init__()

        val = val or train
        test = test or val

        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None

        random.seed(seed)

        paths_source = [
            os.path.join(train.source, fname)
            for fname in os.listdir(train.source)
            if fname.endswith(IMG_EXTS)
        ]
        paths_target = [
            os.path.join(train.target, fname)
            for fname in os.listdir(train.target)
            if fname.endswith(IMG_EXTS)
        ]

        self.train_paths_source = sorted(paths_source)
        self.train_paths_target = sorted(paths_target)

        paths_source = [
            os.path.join(val.source, fname)
            for fname in os.listdir(val.source)
            if fname.endswith(IMG_EXTS)
        ]
        paths_target = [
            os.path.join(val.target, fname)
            for fname in os.listdir(val.target)
            if fname.endswith(IMG_EXTS)
        ]

        self.val_paths_source = sorted(paths_source)
        self.val_paths_target = sorted(paths_target)

        paths_source = [
            os.path.join(test.source, fname)
            for fname in os.listdir(test.source)
            if fname.endswith(IMG_EXTS)
        ]
        paths_target = [
            os.path.join(test.target, fname)
            for fname in os.listdir(test.target)
            if fname.endswith(IMG_EXTS)
        ]

        self.test_paths_source = sorted(paths_source)
        self.test_paths_target = sorted(paths_target)

        self.batch_size = train.batch_size
        self.val_batch_size = val.batch_size
        self.test_batch_size = test.batch_size
        # self.image_p_transform = None
        self.image_p_transform = PairTransform(
            crop_size=CROP, p=0.5, seed=seed
        )
        # self.val_image_p_transform = None
        self.val_image_p_transform = PairTransform(
            crop_size=CROP, p=0.0, seed=seed
        )

        self.image_train_transform = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.image_val_transform = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.image_test_transform = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(
                self.train_paths_source, self.train_paths_target, self.image_train_transform, self.image_p_transform,
            )
            self.val_dataset = Dataset(
                self.val_paths_source, self.val_paths_target, self.image_val_transform, self.val_image_p_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = Dataset(
                self.test_paths_source, self.test_paths_target, self.image_test_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
