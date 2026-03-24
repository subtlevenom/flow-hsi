from typing import Any
from pathlib import Path
from .metrics import register_comparers

comparers: dict = {}


def compare(metrics: str, src: Any, ref: Any, **kwargs):
    """Converts data"""

    fn = comparers.get(metrics, None)
    if fn is None:
        print(f'Comparer "{metrics}" is not found.')
        return None

    try:
        return fn(src, ref, **kwargs)

    except Exception as e:
        print(f'Comparer {metrics} exception: {e}.')
        return None


# register converters

register_comparers(comparers, 'calculate')
