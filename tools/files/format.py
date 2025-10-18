from enum import Enum
from pathlib import Path


class Format(str, Enum):
    MAT = 'mat'
    NPY = 'npy'
    RGB = 'rgb'

    def __str__(self) -> str:
        return self.value


SUFFIX = {
    '.npy': Format.NPY,
    '.mat': Format.MAT,
    '.png': Format.RGB,
    '.bmp': Format.RGB,
    '.jpg': Format.RGB,
    '.jpeg': Format.RGB,
    '.tiff': Format.RGB,
}


def get_format(path):
    return SUFFIX.get(Path(path).suffix, None)
