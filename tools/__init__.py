import numpy as np
from omegaconf import DictConfig
from tools.utils import text
from tools.files import Reader, Writer, Converter


def main(config: DictConfig) -> None:
    npy = Reader().read('/data/korepanov/CAVE/Test/HSI/photo_and_face.mat')
    npy = Converter().convert('hs_rgb', npy, **config.params)
    npy = Converter().convert('denormalize', npy)
    npy = Writer().write('/home/korepanov/work/flow-hsi/.experiments/temp/water.png', npy)
    return None

