import numpy as np
from omegaconf import DictConfig
from .files import convert


def convert(input_path, output_path) -> None:
    match config.tool:
        case 'convert':
            convert(config.params)
    
    npy = Reader().read('/data/korepanov/CAVE/Test/HSI/photo_and_face.mat')
    npy = Converter().convert('hs_rgb', npy, **config.params)
    npy = Converter().convert('normalize', npy, range=[0,255])
    npy = Writer().write('/home/korepanov/work/flow-hsi/.experiments/temp/water.png', npy)
    return None

