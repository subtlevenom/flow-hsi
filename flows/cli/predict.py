import argparse
from pathlib import Path
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
from tools.utils import models
from tools.files.iterators import files
from tools.files import reader
from flows.ml.metrics import (PSNR, SSIM, DeltaE)



def main(config: DictConfig) -> None:
    src_path = Path(config.source_path)
    tgt_path = Path(config.target_path)

    model = ModelSelector.select(config.model).eval()

    checkpoint_path = Path(config.save_dir).joinpath(config.experiment,'logs/checkpoints/last.ckpt')
    models.load_model(model, 'model', checkpoint_path)

    psnr = PSNR(data_range=(0, 1))

    metrics = []

    for src_file in files(src_path):
        tgt_file = tgt_path.joinpath(src_file.name)

        src = reader.read(src_file)
        src = torch.from_numpy(src).to(torch.float32).unsqueeze(0)
        src = src.permute(0,3,1,2)

        tgt = reader.read(tgt_file)
        tgt = torch.from_numpy(tgt).to(torch.float32).unsqueeze(0)
        tgt = tgt.permute(0,3,1,2)
        
        pred = model(image=src)['result']

        val = psnr(pred,tgt)
        metrics.append(val)
        print(f'{src_file.stem}: {val}')

    print(f'AVG: {sum(metrics) / len(metrics)}')

    return 0

    

    
