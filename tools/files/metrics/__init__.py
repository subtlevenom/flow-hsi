import pkgutil
from typing import Callable


def register_comparers(comparers: dict, method: str):
    """registers all converters"""

    def register(name:str, metric:Callable):
        if name in comparers:
            print(f'Metrics "{name}" already exists. Gets replaced.')
        comparers[name] = metric

    for importer, modname, ispkg in pkgutil.iter_modules(__path__):
        module = importer.find_module(modname).load_module(modname)
        register(modname, getattr(module, method))
