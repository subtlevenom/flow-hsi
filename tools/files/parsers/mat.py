from pathlib import Path
import numpy as np
from scipy import io

__FORMATS__ = [
    '.mat',
]

TAG = 'hsi'


def read(file: Path) -> np.ndarray:
    """reads hyperspectral image"""
    return io.loadmat(file)[TAG]


def write(path: Path, image: np.ndarray):
    """writes hyperspectral image"""
    path.parent.mkdir(parents=True, exist_ok=True)
    return io.savemat(path, {TAG: image})
