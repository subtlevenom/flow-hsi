from pathlib import Path
import numpy as np

__FORMATS__ = [
    '.npy',
    '.np',
]


def read(file: Path) -> np.ndarray:
    """reads hyperspectral image"""
    return np.load(file)


def write(path: Path, image: np.ndarray):
    """writes hyperspectral image"""
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.save(path, image)
