from pathlib import Path
import numpy as np
import cv2

__FORMATS__ = [
    '.bmp',
    '.dib',
    '.jpg',
    '.jpeg',
    '.jpe',
    '.jp2',
    '.png',
    '.pbm',
    '.pgm',
    '.ppm',
    '.tiff',
    '.tif',
]


def read(file: Path) -> np.ndarray:
    """reads sRGB image [0,255]"""
    image = cv2.imread(str(file), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.


def write(path: Path, image: np.ndarray, normalize: bool = True):
    """Writes sRGB image"""
    image = np.clip(255. * image, a_min=0., a_max=255.)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.imwrite(str(path), image)
