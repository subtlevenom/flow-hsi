from pathlib import Path
import importlib
from typing import Callable


def register_task(task) -> Callable:
    """Gets entry point for task"""

    module_path = list(Path(__file__).parent.glob(f'{task}.py'))
    if len(module_path) == 1:
        module = importlib.import_module('.' + module_path[0].stem, __name__)
        if getattr(module, 'main', None) is not None:
            return module.main

    raise ValueError(f'Module for task {task} not found.')
