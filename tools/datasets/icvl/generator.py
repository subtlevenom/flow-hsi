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


def generate(input_path: str, params: DictConfig, optics: DictConfig, **kwargs) -> None:
    """
    CAVE-HSI: https://ieee-dataport.org/documents/cave-hsi
    """

    input_dir = Path(input_path)

    if not input_dir.is_dir():
        raise Exception(f'No such directory: {input_dir}')

    cdf = create_cdf(optics)

    for gt_path in Path(input_dir).glob('**/*.mat'):
        try:
            src_path = gt_path.parent.parent.joinpath(params.tag).joinpath(gt_path.name)
            gt_image = reader.read(gt_path)
            src_image = cdf(gt_image, padding=params.padding)
            # normalize
            src_max, src_min = src_image.max(), src_image.min()
            src_image = (src_image - src_min) / (src_max - src_min) # * 65535).clip(0, 65535).astype(np.uint16)
            writer.write(src_path, src_image)
        except Exception as e:
            print(f'Failed to convert {gt_path}.')
