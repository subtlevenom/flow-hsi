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
    center_crop = A.CenterCrop(height=512, width=512)

    n = 0
    src = []
    gt = []
    for gt_path in Path(input_dir).glob('**/*.mat'):
        try:
            src_path = gt_path.parent.parent.joinpath(params.tag).joinpath(
                gt_path.name)
            gt_image = reader.read(gt_path)
            src_image = cdf(gt_image, padding=256)
            # normalize
            mx_gt = np.mean(gt_image, axis=(0,1), keepdims=True)
            mx_src = np.mean(src_image, axis=(0,1), keepdims=True)
            src.append(mx_src)
            gt.append(mx_gt)
            n += 1
        except Exception as e:
            print(f'Failed to convert {gt_path}.')

    src = sum(src) / n
    gt = sum(gt) / n
    mx = gt / src

    for gt_path in Path(input_dir).glob('**/*.mat'):
        try:
            src_path = gt_path.parent.parent.joinpath(params.tag).joinpath(
                gt_path.name)
            gt_image = reader.read(gt_path)
            src_image = cdf(gt_image, padding=256)
            # normalize
            src_image = src_image * mx 
            writer.write(src_path, src_image)
        except Exception as e:
            print(f'Failed to convert {gt_path}.')
