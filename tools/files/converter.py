from typing import Any
from pathlib import Path
from .converters import register_converters

converters: dict = {}


def convert(converter_name: str, data: Any, **kwargs):
    """Converts data"""

    converter = converters.get(converter_name, None)
    if converter is None:
        print(f'Converter "{converter_name}" is not found.')
        return None

    try:
        return converter(data, **kwargs)

    except Exception as e:
        print(f'Format {format} write exception: {e}.')
        return None


# register converters

register_converters(converters, 'convert')
