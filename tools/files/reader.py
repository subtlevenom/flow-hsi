import os
import glob
import logging
from pathlib import Path
import pandas as pd
from typing import Callable
from .format import Format, get_format

logger = logging.getLogger(__name__)


class Reader:
    """File reader"""

    parsers: dict = {}

    @classmethod
    def register_parser(cls, format: Format, parser: Callable):
        """Registers external parser"""

        if format in cls.parsers:
            logger.info(f'Parser {format} already exists. Gets replaced.')
        cls.parsers[format] = parser

    # public

    def read(self, path:str):
        """reads file(s) in path"""

        path:Path = Path(path)

        if path.is_dir():
            return self.read_dir(path)
        else:
            return self.read_file(path)

    # private

    def read_dir(self, path: Path, format: Format):
        """Reads dir"""

        data = [self.read_file(f) for f in path.glob('**/*') if f.is_file()]
        return filter(lambda v: v is not None, data)

    def read_file(self, path:Path):
        """Reads file of given format"""

        if not path.exists():
            logger.error(f'File {path} not found.')
            return None

        format = get_format(path)
        if format is None:
            logger.error(f'Unknown format of {path}.')
            return None

        parser = self.parsers.get(format, None)
        if parser is None:
            logger.error(f'Reader for format {format} is not found.')
            return None

        try:
            return parser(path)

        except Exception as e:
            logger.error(f'Format {format} read exception: {e}.')
            return None
