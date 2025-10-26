from typing import List, Tuple
import numpy as np
import cv2
from tools.utils import metrics


def calculate(src: np.ndarray, ref: np.ndarray, **kwargs) -> float:
    """psnr"""
    src /= np.max(src)
    ref /= np.max(ref)
    return metrics.psnr(src, ref)

    
