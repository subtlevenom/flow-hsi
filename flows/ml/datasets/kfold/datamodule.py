import os
import random
import torch
from pathlib import Path
import lightning as L
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    Normalize,
    ToDtype,
)
from torch.utils.data import DataLoader, ConcatDataset
from typing import Tuple
from flows.core import Logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset


class KFoldDataModule(L.LightningDataModule):

    def __init__(
        self,
        dm: L.LightningDataModule,
        folds: int,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.dm = dm
        self.kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    def __iter__(self):
        self.dm.setup('fit')
        train_dataset = self.dm.train_dataloader().dataset
        val_dataset = self.dm.val_dataloader().dataset
        self.dm.setup('test')
        test_dataset = self.dm.test_dataloader().dataset
        self.dataset = ConcatDataset(
            [train_dataset, val_dataset, test_dataset])

        for fold, (train_idx,
                   val_idx) in enumerate(self.kf.split(self.dataset)):
            self.train_idx = train_idx
            self.val_idx = val_idx
            yield self

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = Subset(self.dataset, self.train_idx)
            self.val_dataset = Subset(self.dataset, self.val_idx)
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset

    def train_dataloader(self) -> DataLoader:
        dl: DataLoader = self.dm.train_dataloader()
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=dl.batch_size,
            shuffle=True,
            num_workers=dl.num_workers,
            pin_memory=dl.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        dl: DataLoader = self.dm.val_dataloader()
        return DataLoader(
            self.val_dataset,
            batch_size=dl.batch_size,
            shuffle=False,
            num_workers=dl.num_workers,
            pin_memory=dl.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        dl: DataLoader = self.dm.test_dataloader()
        return DataLoader(
            self.test_dataset,
            batch_size=dl.batch_size,
            shuffle=False,
            num_workers=dl.num_workers,
            pin_memory=dl.pin_memory,
        )
