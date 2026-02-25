import os
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from rich.progress import Progress
import cv2
import random
import numpy as np
import imageio
import albumentations as A
import shutil
from omegaconf import DictConfig
from tools.utils.concurrent import concurrent
from tools.utils import images
from tools.files import reader
from scipy import io
from tools.files.iterators import files

THREADS = 1

SOURCE = 'source'
TARGET = 'target'


def split(source_path:str, target_path:str, output_path:str, split:dict) -> None:
    """
    ICVL: 
    """

    source_dir = Path(source_path)
    target_dir = Path(target_path)
    output_dir = Path(output_path)

    if not source_dir.is_dir():
        raise Exception(f'No such directory: {source_dir}')
    if not target_dir.is_dir():
        raise Exception(f'No such directory: {source_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = set([f.name for f in files(source_dir, '.npy')])
    target_files = set([f.name for f in files(target_dir, '.npy')])
    common_files = list(source_files & target_files)

    # split
    random.shuffle(common_files)
    split_slices = np.cumsum([int(s * len(common_files)) for s in split.values()])
    split_files = np.split(common_files, split_slices[:-1])

    with Progress() as progress:
        for name, sfiles in zip(split.keys(), split_files):
            pb = progress.add_task(f"[cyan]{name}", total=2*len(sfiles))
            try:
                # source
                output_split_dir = output_dir.joinpath(name, SOURCE)
                output_split_dir.mkdir(parents=True, exist_ok=True)
                for file in sfiles:
                    input_path = source_dir.joinpath(file)
                    output_path = output_split_dir.joinpath(file)
                    shutil.copy(input_path, output_path)
                    progress.update(pb, advance=1)
                # target
                output_split_dir = output_dir.joinpath(name, TARGET)
                output_split_dir.mkdir(parents=True, exist_ok=True)
                for file in sfiles:
                    input_path = target_dir.joinpath(file)
                    output_path = output_split_dir.joinpath(file)
                    shutil.copy(input_path, output_path)
                    progress.update(pb, advance=1)

            except Exception as e:
                print(f'Skip split {name} with exception {e}')
