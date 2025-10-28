import argparse
import yaml
from omegaconf import DictConfig
from ..core.selector import (ModelSelector, DataSelector, PipelineSelector)
import lightning as L
import os
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from flows.ml.callbacks import GenerateCallback
from lightning.pytorch.loggers import CSVLogger
from flows import cli
from tools.utils import text


def main(config: DictConfig) -> None:
    text.print(config)

    dm = DataSelector.select(config.data)
    model = ModelSelector.select(config.model)
    pipeline = PipelineSelector.select(model, config.pipeline)

    logger = CSVLogger(
        save_dir=os.path.join(config.save_dir, config.experiment),
        name='logs',
        version='',
    )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=os.path.join(config.save_dir, config.experiment),
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        devices='auto',
        callbacks=[
            RichProgressBar(),
            GenerateCallback(
                every_n_epochs=1,
            ),
        ],
    )

    ckpt_path = os.path.join(config.save_dir, config.experiment, 'logs/checkpoints/last.ckpt')

    if not os.path.exists(ckpt_path):
        ckpt_path = None

    trainer.test(
        model=pipeline, 
        datamodule=dm,
        ckpt_path=ckpt_path,
    )
