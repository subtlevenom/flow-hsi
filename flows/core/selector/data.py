
from omegaconf import DictConfig
from flows.ml.datasets.kfold.datamodule import KFoldDataModule
from typing import Union
from lightning import LightningDataModule
from flows.ml.datasets import create_dataset


class DataSelector:

    def select(config: DictConfig) -> Union[LightningDataModule]:
        
        datamodule = create_dataset(config.name, config.params)    
        
        # N-folds data splitting
        if config.get('folds', 1) > 1:
            datamodule = KFoldDataModule(datamodule, config.folds)
        
        return datamodule

