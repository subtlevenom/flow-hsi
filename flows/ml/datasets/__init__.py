import sys, inspect
import lightning as L
from . import kfold, sr_scale, cave_hsi


def create_dataset(name, params) -> L.LightningDataModule:
    params = params or {}
    modules = sys.modules[__name__]
    module = getattr(modules, name)
    for name, cls in inspect.getmembers(module):
        if inspect.isclass(cls) and issubclass(cls, L.LightningDataModule):
            return cls(**params)
    return None
