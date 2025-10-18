import pkgutil
from typing import Callable, List


def register_parsers(parsers: dict, method: str):
    """registers all parsers"""

    def register(formats: List[str], parser: Callable):
        for suffix in formats:
            if suffix in parsers:
                print(f'Parser "{suffix}" already exists. Gets replaced.')
            parsers[suffix] = parser

    for importer, modname, ispkg in pkgutil.iter_modules(__path__):
        module = importer.find_module(modname).load_module(modname)
        formats = getattr(module, '__FORMATS__')
        register(formats, getattr(module, method))
