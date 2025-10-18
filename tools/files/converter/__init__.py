import importlib
import inspect
import pkgutil

from omegaconf import DictConfig
from . import converters
from .converter import Converter


def register_parsers():
    """registers all parsers"""

    for importer, modname, ispkg in pkgutil.iter_modules(parsers.__path__):
        module = importer.find_module(modname).load_module(modname)
        formats = getattr(module, '__FORMATS__')
        Reader.register_parser(formats, getattr(module, 'read'))
        Writer.register_parser(formats, getattr(module, 'write'))


def register_converters():
    """registers all converters"""

    for importer, modname, ispkg in pkgutil.iter_modules(converters.__path__):
        module = importer.find_module(modname).load_module(modname)
        Converter.register_converter(modname, getattr(module, 'convert'))


register_parsers()
register_converters()

# 

def convert(config: DictConfig) -> None:
    npy = Reader().read('/data/korepanov/CAVE/Test/HSI/photo_and_face.mat')
    npy = Converter().convert('hs_rgb', npy, **config.params)
    npy = Converter().convert('normalize', npy, range=[0,255])
    npy = Writer().write('/home/korepanov/work/flow-hsi/.experiments/temp/water.png', npy)
    return None
