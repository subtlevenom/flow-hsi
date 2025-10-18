from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy import io
from tools.utils.hsi import rgb_to_hs


def convert(image: np.ndarray, channels: int) -> np.ndarray:
    """converts rgb to hyperspectral image"""
    return rgb_to_hs(image, channels)
