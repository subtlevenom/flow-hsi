import os
import shutil
import logging
from typing import Any
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class Converter:
    """Converter"""

    converters:dict = {}

    @classmethod
    def register_converter(cls, name, converter):
        """Registers external parser"""

        if name in cls.converters:
            logger.warning(f'Parser "{name}" already exists. Gets replaced.')

        cls.converters[name] = converter

    def convert(self, converter_name: str, data: Any, **kwargs):
        """Converts data"""

        converter = self.converters.get(converter_name, None)
        if converter is None:
            logger.error(f'Converter "{converter_name}" is not found.')
            return None

        try:
            return converter(data, **kwargs)

        except Exception as e:
            logger.error(f'Format {format} write exception: {e}.')
            return None
