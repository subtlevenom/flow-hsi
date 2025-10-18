from pathlib import Path
import pkgutil
from typing import Any, Callable, List
from .parsers import register_parsers

parsers: dict = {}


def read(path: str | Path) -> Any:
    """Reads file"""

    path = Path(path)

    if not path.exists():
        print(f'File {path} not found.')
        return None

    parser = parsers.get(path.suffix, None)
    if parser is None:
        print(f'Reader for format "{path.suffix}" is not found.')
        return None

    try:
        return parser(path)

    except Exception as e:
        print(f'Format {format} read exception: {e}.')
        return None


# register parsers

register_parsers(parsers, 'read')
