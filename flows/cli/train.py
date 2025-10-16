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

    if config.data.get('folds', 1) > 1:
        train_kfold(config)
    else:
        train_default(config)


def train_default(config: DictConfig) -> None:

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
                filename="{epoch}-{val_de:.2f}",
                monitor='val_psnr',
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

    trainer.fit(
        model=pipeline,
        datamodule=dm,
        ckpt_path=ckpt_path
        if config.resume and os.path.exists(ckpt_path) else None,
    )


def train_kfold(config: DictConfig) -> None:

    FOLD_STEPS = 10

    dm = DataSelector.select(config.data)
    model = ModelSelector.select(config.model)
    pipeline = PipelineSelector.select(model, config.pipeline)

    logger = CSVLogger(
        save_dir=os.path.join(config.save_dir, config.experiment),
        name='logs',
        version='',
    )

    ckpt_path = os.path.join(config.save_dir, config.experiment,
                             'logs/checkpoints/last.ckpt')
    resume = config.resume and os.path.exists(ckpt_path)
    if resume:
        state_dict = torch.load(ckpt_path)
        current_epoch = state_dict['epoch']
    else:
        ckpt_path = None
        current_epoch = 0

    while current_epoch < config.epochs:
        for fold, data_module in enumerate(dm):
            print(f'Fold {fold + 1}')

            trainer = L.Trainer(
                logger=logger,
                default_root_dir=os.path.join(config.save_dir,
                                              config.experiment),
                max_epochs=current_epoch + FOLD_STEPS,
                devices=1,
                callbacks=[
                    ModelCheckpoint(
                        filename="{epoch}-{val_loss:.2f}",
                        monitor='val_de',
                        save_top_k=3,
                        save_last=True,
                    ),
                    RichModelSummary(),
                    RichProgressBar(),
                    LearningRateMonitor(logging_interval='epoch', ),
                    GenerateCallback(every_n_epochs=1, ),
                    # LearningRateCallback(num_training_steps=100),
                    StochasticWeightAveraging(
                        swa_lrs=config.pipeline.params.lr * 10.)
                ],
            )

            trainer.fit(
                model=pipeline,
                datamodule=data_module,
                ckpt_path=ckpt_path,
            )

            current_epoch = trainer.current_epoch
            resume = True
