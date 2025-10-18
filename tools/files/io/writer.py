import logging
from typing import Any, Callable, List
from pathlib import Path

logger = logging.getLogger(__name__)


class Writer:
    """Writer"""

    parsers: dict = {}

    @classmethod
    def register_parser(cls, formats: List[str], parser: Callable):
        """Registers external parser"""

        for suffix in formats:
            if suffix in cls.parsers:
                logger.warning(
                    f'Parser "{suffix}" already exists. Gets replaced.')
            cls.parsers[suffix] = parser

    @classmethod
    def write(cls, path: str, data: Any):
        """Writer data to file"""

        path: Path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            logger.info(f'File {path} already exists. Gets replaced.')

        parser = cls.parsers.get(path.suffix, None)
        if parser is None:
            logger.error(f'Writer for format {format} is not found.')
            return None

        try:
            return parser(path, data)

        except Exception as e:
            logger.error(f'Format {format} write exception: {e}.')
            return None
