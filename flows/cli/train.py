import argparse
import yaml
from omegaconf import DictConfig, OmegaConf
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
from lightning.pytorch.loggers import CSVLogger
from flows.ml.callbacks import GenerateCallback
from flows.tools.utils import text


def main(config: DictConfig) -> None:
    text.print_config(config)

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

    # cmKAN-Killer Training Configuration
    trainer = L.Trainer(
        logger=logger,
        default_root_dir=os.path.join(config.save_dir, config.experiment),
        max_epochs=config.epochs,
        # Use bf16 for better numerical stability in Gaussian kernels
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 32,
        devices=1,
        # Tighter clipping for Quadratic Transport stability
        gradient_clip_val=0.5,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{val_mrae:.4f}",
                monitor='val_mrae',  # NTIRE spectral primary metric
                save_top_k=3,
                save_last=True,
                mode='min',
            ),
            RichModelSummary(),
            RichProgressBar(),
            LearningRateMonitor(logging_interval='step'),
            GenerateCallback(every_n_epochs=5),  # Reduce frequency to save IO
            StochasticWeightAveraging(
                # SWA LR should be low to smooth the transport manifold
                swa_lrs=config.pipeline.params.lr * 0.1,
                swa_epoch_start=int(0.75 * config.epochs),
            )
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
