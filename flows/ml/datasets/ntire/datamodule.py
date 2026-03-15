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

IMG_EXTS: Tuple[str] = (".npy")


class DataModule(L.LightningDataModule):

    def __init__(
        self,
        train: dict,
        val: dict = None,
        test: dict = None,
        predict: dict = None,
        num_workers: int = min(12,
                               os.cpu_count() - 2),
        seed: int = 43,
        **kwargs,
    ) -> None:
        super().__init__()

        val = val or train
        test = test or val
        predict = predict or test

        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

        random.seed(seed)

        # train
        paths_source = [
            os.path.join(train.source, fname)
            for fname in os.listdir(train.source) if fname.endswith(IMG_EXTS)
        ]
        paths_target = [
            os.path.join(train.target, fname)
            for fname in os.listdir(train.target) if fname.endswith(IMG_EXTS)
        ]
        self.train_paths_source = sorted(paths_source)
        self.train_paths_target = sorted(paths_target)
        # val
        paths_source = [
            os.path.join(val.source, fname) for fname in os.listdir(val.source)
            if fname.endswith(IMG_EXTS)
        ]
        paths_target = [
            os.path.join(val.target, fname) for fname in os.listdir(val.target)
            if fname.endswith(IMG_EXTS)
        ]
        self.val_paths_source = sorted(paths_source)
        self.val_paths_target = sorted(paths_target)
        # test
        paths_source = [
            os.path.join(test.source, fname)
            for fname in os.listdir(test.source) if fname.endswith(IMG_EXTS)
        ]
        paths_target = [
            os.path.join(test.target, fname)
            for fname in os.listdir(test.target) if fname.endswith(IMG_EXTS)
        ]
        self.test_paths_source = sorted(paths_source)
        self.test_paths_target = sorted(paths_target)
        # predict
        paths_source = [
            os.path.join(predict.source, fname)
            for fname in os.listdir(predict.source) if fname.endswith(IMG_EXTS)
        ]
        paths_target = [
            os.path.join(predict.target, fname)
            for fname in os.listdir(predict.target) if fname.endswith(IMG_EXTS)
        ]
        self.predict_paths_source = sorted(paths_source)
        self.predict_paths_target = sorted(paths_target)
        # train
        self.train_batch_size = train.batch_size
        self.train_crop_size = train.get('crop_size',0)
        self.train_norm = train.get('norm',1.0)
        # val
        self.val_batch_size = val.batch_size
        self.val_crop_size = val.get('crop_size',0)
        self.val_norm = train.get('norm',1.0)
        # test
        self.test_batch_size = test.batch_size
        self.test_crop_size = test.get('crop_size',0)
        self.test_norm = train.get('norm',1.0)
        # predict
        self.predict_batch_size = predict.batch_size
        self.predict_crop_size = predict.get('crop_size',0)
        self.predict_norm = train.get('norm',1.0)
        # pair transforms
        self.train_image_p_transform = PairTransform(
            crop_size=self.train_crop_size,
            p=0.5,
            seed=seed,
        )
        self.val_image_p_transform = PairTransform(
            crop_size=self.val_crop_size,
            p=0.0,
            seed=seed,
        )
        self.test_image_p_transform = PairTransform(
            crop_size=self.test_crop_size,
            p=0.0,
            seed=seed,
        )
        self.predict_image_p_transform = PairTransform(
            crop_size=self.predict_crop_size,
            p=0.0,
            seed=seed,
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
        self.image_predict_transform = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(
                self.train_paths_source,
                self.train_paths_target,
                self.image_train_transform,
                self.train_image_p_transform,
                norm=self.train_norm,
            )
            self.val_dataset = Dataset(
                self.val_paths_source,
                self.val_paths_target,
                self.image_val_transform,
                self.val_image_p_transform,
                norm=self.val_norm,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = Dataset(
                self.test_paths_source,
                self.test_paths_target,
                self.image_test_transform,
                self.test_image_p_transform,
                norm=self.test_norm,
            )
        if stage == 'predict' or stage is None:
            self.predict_dataset = Dataset(
                self.predict_paths_source,
                self.predict_paths_target,
                self.image_predict_transform,
                self.predict_image_p_transform,
                norm=self.predict_norm,
                filename=True,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
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

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
