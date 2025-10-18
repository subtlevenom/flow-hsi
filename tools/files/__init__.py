import importlib
import inspect
import pkgutil 
from . import parsers
from . import converters
from .format import Format
from .reader import Reader
from .writer import Writer
from .converter import Converter


def register_parsers():
    """registers all parsers"""

    for format in Format:
        module = importlib.import_module(
            name='.' + str(format),
            package=parsers.__package__,
        )
        Reader.register_parser(format, getattr(module, 'read'))
        Writer.register_parser(format, getattr(module, 'write'))


def register_converters():
    """registers all converters"""

    for importer, modname, ispkg in pkgutil.iter_modules(converters.__path__):
        module = importer.find_module(modname).load_module(modname)
        Converter.register_converter(modname, getattr(module, 'convert'))


register_converters()
register_parsers()
