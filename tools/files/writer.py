import os
import shutil
import logging
from typing import Any
import pandas as pd
from pathlib import Path
from .format import Format, get_format

logger = logging.getLogger(__name__)


class Writer:
    """Writer"""

    parsers: dict = {}

    @classmethod
    def register_parser(cls, format, parser):
        """Registers external parser"""

        if format in cls.parsers:
            logger.warning(f'Parser {format} already exists. Gets replaced.')

        cls.parsers[format] = parser

    def write(self, path: str, data: Any):
        """Writer data to file"""

        path: Path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            logger.info(f'File {path} already exists. Gets replaced.')

        format = get_format(path)
        if format is None:
            logger.error(f'Unknown format of {path}.')
            return None

        parser = self.parsers.get(format, None)
        if parser is None:
            logger.error(f'Writer for format {format} is not found.')
            return None

        try:
            return parser(path, data)

        except Exception as e:
            logger.error(f'Format {format} write exception: {e}.')
            return None
