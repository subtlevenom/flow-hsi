import logging
from pathlib import Path
from typing import Any, Callable, List

logger = logging.getLogger(__name__)


class Reader:
    """File reader"""

    parsers: dict = {}

    @classmethod
    def register_parser(cls, formats: List[str], parser: Callable):
        """Registers external parser"""

        for suffix in formats:
            if suffix in cls.parsers:
                logger.info(
                    f'Parser "{suffix}" already exists. Gets replaced.')
            cls.parsers[suffix] = parser

    # public

    def read(self, path: str | Path) -> Any:
        """reads file(s) in path"""

        path = Path(path)

        if path.is_dir():
            return self.read_dir(path)
        else:
            return self.read_file(path)

    def read_dir(self, path: Path) -> List[Any]:
        """Reads dir"""

        data = [self.read_file(f) for f in path.glob('**/*') if f.is_file()]
        return filter(lambda v: v is not None, data)

    def read_file(self, path: Path) -> Any:
        """Reads file of given format"""

        if not path.exists():
            logger.error(f'File {path} not found.')
            return None

        parser = self.parsers.get(path.suffix, None)
        if parser is None:
            logger.error(f'Reader for format "{path.suffix}" is not found.')
            return None

        try:
            return parser(path)

        except Exception as e:
            logger.error(f'Format {format} read exception: {e}.')
            return None
