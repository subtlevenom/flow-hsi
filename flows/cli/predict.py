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
    predict_default(config)


def predict_default(config: DictConfig) -> None:

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
        devices=1,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{val_loss:.2f}",
                monitor='val_loss',
                save_last=True,
            ),
            RichModelSummary(),
            RichProgressBar(),
            LearningRateMonitor(logging_interval='epoch', ),
            GenerateCallback(every_n_epochs=1, ),
            StochasticWeightAveraging(swa_lrs=config.pipeline.params.lr * 10.)
        ],
    )

    ckpt_path = os.path.join(config.save_dir, config.experiment,
                             'logs/checkpoints/last.ckpt')

    trainer.predict(
        model=pipeline,
        datamodule=dm,
        ckpt_path=ckpt_path
        if config.resume and os.path.exists(ckpt_path) else None,
    )
