import numpy as np
from tools.utils.spectral import rgb_to_hs


def convert(image: np.ndarray, bands: int, **kwargs) -> np.ndarray:
    """converts rgb to hyperspectral image"""
    return rgb_to_hs(image, bands)
