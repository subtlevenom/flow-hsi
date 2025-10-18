import importlib
import inspect
import pkgutil

from omegaconf import DictConfig
from .io import Reader, Writer
from .converter import Converter


reader = Reader()
writer = Writer()
converter = Converter()

def convert(converter, input_path, output_path, format, **kwargs) -> None:
    npy = Reader().read('/data/korepanov/CAVE/Test/HSI/photo_and_face.mat')
    npy = Converter().convert('hs_rgb', npy, **kwargs)
    npy = Converter().convert('normalize', npy, range=[0,255])
    npy = Writer().write('/home/korepanov/work/flow-hsi/.experiments/temp/water.png', npy)
    return None
