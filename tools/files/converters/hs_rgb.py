from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy import io
from tools.utils.hsi import hs_to_rgb


def convert(image: np.ndarray, channels: List[Tuple[int]]) -> np.ndarray:
    """converts hyperspectral image to rgb"""
    return hs_to_rgb(image, channels)
