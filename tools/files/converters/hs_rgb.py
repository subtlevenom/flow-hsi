from typing import List, Tuple
import numpy as np
import cv2
from tools.utils.spectral import hs_to_rgb


def convert(image: np.ndarray, bands: List[Tuple[int]], **kwargs) -> np.ndarray:
    """converts hyperspectral image to rgb"""
    rgb = hs_to_rgb(image, bands)
    return rgb / np.max(rgb)

    
