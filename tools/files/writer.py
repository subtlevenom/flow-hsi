from typing import Any, Callable, List
from pathlib import Path
from .parsers import register_parsers

parsers: dict = {}


def write(path: str | Path, data: Any, format:str = None, **kwargs):
    """Writer data to file"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f'File {path} already exists. Gets replaced.')

    parser = parsers.get(format or path.suffix, None)
    if parser is None:
        print(f'Writer for format {format} is not found.')
        return None

    try:
        return parser(path, data)

    except Exception as e:
        print(f'Format {format} write exception: {e}.')
        return None


# register parsers

register_parsers(parsers, 'write')
