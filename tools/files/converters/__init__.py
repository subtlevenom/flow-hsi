import pkgutil
from typing import Callable


def register_converters(converters: dict, method: str):
    """registers all converters"""

    def register(name:str, converter:Callable):
        if name in converters:
            print(f'Parser "{name}" already exists. Gets replaced.')
        converters[name] = converter

    for importer, modname, ispkg in pkgutil.iter_modules(__path__):
        module = importer.find_module(modname).load_module(modname)
        register(modname, getattr(module, method))
