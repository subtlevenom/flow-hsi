import argparse
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
from rich.progress import Progress
import yaml
from gazenet.video_tracker.video_tracker import VideoTracker
from openeye import cli
from datasets.odc import Odc
from openeye.core.config.config import Config
from openeye.core.selector.model import ModelSelector
from ..ml.utils.models import load_model



def main(config: DictConfig) -> None:
    input_path = Path(config.input)
    output_path = Path(config.output)

    model = ModelSelector.select(config.model).eval()

    # checkpoint_path = '.experiments/resnet.se.fusion.attention/logs/checkpoints/epoch=111-val_loss=0.01.ckpt'
    checkpoint_path = Path(config.save_dir).joinpath(config.experiment,'logs/checkpoints/last.ckpt')
    load_model(model, 'model', checkpoint_path)

    tracker = VideoTracker(model)
    tracker(input_path, output_path)

    return 0

    

    
