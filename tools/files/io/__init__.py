import pkgutil
from . import parsers
from .reader import Reader
from .writer import Writer


def register_parsers():
    """registers all parsers"""

    for importer, modname, ispkg in pkgutil.iter_modules(parsers.__path__):
        module = importer.find_module(modname).load_module(modname)
        formats = getattr(module, '__FORMATS__')
        Reader.register_parser(formats, getattr(module, 'read'))
        Writer.register_parser(formats, getattr(module, 'write'))


register_parsers()