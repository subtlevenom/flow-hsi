import numpy as np
from tools.utils.hsi import rgb_to_hs


def convert(image: np.ndarray, channels: int, **kwargs) -> np.ndarray:
    """converts rgb to hyperspectral image"""
    return rgb_to_hs(image, channels)
