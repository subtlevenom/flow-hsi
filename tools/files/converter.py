from typing import Any
from pathlib import Path
from .converters import register_converters

converters: dict = {}


def convert(converter: str, data: Any, **kwargs):
    """Converts data"""

    fn = converters.get(converter, None)
    if fn is None:
        print(f'Converter "{converter}" is not found.')
        return None

    try:
        return fn(data, **kwargs)

    except Exception as e:
        print(f'Converter {converter} exception: {e}.')
        return None


# register converters

register_converters(converters, 'convert')
