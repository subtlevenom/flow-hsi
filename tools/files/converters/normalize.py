import numpy as np
from tools.utils.images import normalize


def convert(image: np.ndarray, **kwargs) -> np.ndarray:
    """converts rgb to rgb [0,255]"""
    return normalize(image)
