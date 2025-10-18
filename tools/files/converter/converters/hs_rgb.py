from typing import List, Tuple
import numpy as np
from tools.utils.hsi import hs_to_rgb


def convert(image: np.ndarray, channels: List[Tuple[int]], **kwargs) -> np.ndarray:
    """converts hyperspectral image to rgb"""
    return hs_to_rgb(image, channels)
