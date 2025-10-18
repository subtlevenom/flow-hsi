from typing import List, Tuple
import cv2 as cv
import numpy as np
from scipy.signal.windows import gaussian
# np.set_printoptions(threshold=sys.maxsize)


def hs_to_rgb(hs_image: np.ndarray, channels: List[Tuple[int]]):
    """
    input: hs image normalized [0,1]
    output: rgb image normalized [0,1]
    """
    rgb = []
    for r in channels:
        len = r[1] - r[0]
        window = gaussian(len, std=np.sqrt(len))
        signal = hs_image[:, :, r[0]:r[1]]
        signal = np.multiply(signal, window) / np.sum(window)
        signal = np.sum(signal, axis=1, keepdims=True)
        rgb.append(signal)
    rgb = np.concat(rgb, axis=-1)
    return rgb


def rgb_to_hs(image: np.ndarray):
    """xyz input image"""
    pass
    # sRGB = colour.RGB_COLOURSPACES["sRGB"]
    # return sRGB.cctf_encoding(image)
