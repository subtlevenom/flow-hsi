from pathlib import Path
import numpy as np
from scipy import io
import h5py

__FORMATS__ = [
    'icvl',
]

TAG = 'rad'


def read(file: Path) -> np.ndarray:
    """reads hyperspectral image"""
    with h5py.File(file, 'r') as mat:
        mat = np.array(mat[TAG])
    mat = np.transpose(mat) #.moveaxis(mat, 0, 2)
    return mat


def write(path: Path, image: np.ndarray):
    """writes hyperspectral image"""
    raise NotImplementedError()
