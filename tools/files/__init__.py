import importlib
import inspect
import pkgutil
from . import parsers
from . import converters
from .reader import Reader
from .writer import Writer
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
