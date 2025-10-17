import importlib
from . import parsers
from .format import Format
from .reader import Reader
from .writer import Writer


def register():
    """registers all parsers"""

    for format in Format:
        module = importlib.import_module(str(format), package=parsers)
        Reader.register_parser(format, getattr(module, 'read'))
        Writer.register_parser(format, getattr(module, 'write'))

register()