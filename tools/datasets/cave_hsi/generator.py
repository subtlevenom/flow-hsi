from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from rich.progress import Progress
import cv2
import random
import numpy as np
import imageio
from scipy import io
import albumentations as A
from omegaconf import DictConfig
from tools.utils.concurrent import concurrent
from tools.optics import create_cdf
from tools.files import reader, writer


def generate(input_path: str, output_path:str, **kwargs) -> None:
    """
    CAVE-HSI: https://ieee-dataport.org/documents/cave-hsi
    """

    input_dir = Path(input_path)
    if not input_dir.is_dir():
        raise Exception(f'No such directory: {input_dir}')

    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    for gt_path in Path(input_dir).glob('**/*.mat'):
        try:
            gt_image = reader.read(gt_path)
            out_path = output_dir.joinpath(gt_path.stem).with_suffix('.npy')
            writer.write(out_path, gt_image)
        except Exception as e:
            print(f'Failed to convert {gt_path}.')
