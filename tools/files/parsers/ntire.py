from pathlib import Path
import numpy as np
from scipy import io

__FORMATS__ = [
    'ntire',
]

TAG = 'hsi'
TAGS = ['cube', 'hsi', 'ref', 'rad']


def read(file: Path) -> np.ndarray:
    """reads hyperspectral image"""
    m = io.loadmat(file)
    for tag in TAGS: 
        data = m.get(tag, None) 
        if data is not None:
            return data


def write(path: Path, image: np.ndarray):
    """writes hyperspectral image"""
    path.parent.mkdir(parents=True, exist_ok=True)
    return io.savemat(path, {TAG: image})
