from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from rich.progress import Progress
import cv2
import random
import numpy as np
import imageio
import albumentations as A
from omegaconf import DictConfig
from tools.utils.concurrent import concurrent
from tools.utils import images
from tools.files import reader
from scipy import io

THREADS = 1

MAT_TYPE = 'icvl'
HSI_SUFFICES = {'.mat','.npy'}
INPUT = 'input'
OUTPUT = 'output'


def generate(split:dict, params:DictConfig, **kwargs) -> None:
    """
    ICVL: 
    """

    common_filestems = None
    for name, data in split.items():
        input_files = [f for f in Path(data.input).glob('*.*') if f.suffix in HSI_SUFFICES]
        input_set = set([f.stem.replace('_gyper','') for f in input_files])
        if common_filestems is None:
            common_filestems = input_set
        else:
            common_filestems = common_filestems & input_set

    # Copy files, convert to npy
    for name, data in split.items():
        input_files = [f for f in Path(data[INPUT]).glob('*.*') if f.suffix in HSI_SUFFICES and f.stem.replace('_gyper','') in common_filestems]
        output_dir = Path(data[OUTPUT])

        for file in input_files:
            try:
                if file.suffix == '.mat':
                    hsi = reader.read(file, MAT_TYPE)
                else:
                    hsi = reader.read(file)
                output_path = output_dir.joinpath(file.name.replace('_gyper','')).with_suffix('.npy')
                np.save(output_path, hsi)

            except Exception as e:
                print(e)
