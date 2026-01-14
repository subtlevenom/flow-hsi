from pathlib import Path
import numpy as np
from scipy import io
import h5py

__FORMATS__ = [
    'ntire',
]

TAG = 'hsi'
TAGS = ['cube', 'hsi', 'ref', 'rad']


def read(file: Path) -> np.ndarray:
    """reads hyperspectral image"""
    with h5py.File(file, 'r') as mat:
        for tag in TAGS: 
            data = mat.get(tag, None) 
            if data is not None:
                data = np.array(data)
                data = np.transpose(data) #.moveaxis(mat, 0, 2)
                return data
    return None


def write(path: Path, image: np.ndarray):
    """writes hyperspectral image"""
    path.parent.mkdir(parents=True, exist_ok=True)
    return io.savemat(path, {TAG: image})
