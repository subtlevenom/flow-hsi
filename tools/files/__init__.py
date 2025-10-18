from pathlib import Path
from typing import Any, List
from omegaconf import DictConfig
from .iterators import files
from . import reader, writer, converter


def convert(
    input_path: str,
    output_path: str,
    suffix: str,
    params: DictConfig,
) -> None:
    """
    input_path - file or dir
    output_path - output dir
    suffix - output file extention
    params - converter dependent args.
    """

    input_path: Path = Path(input_path)
    output_path: Path = Path(output_path)

    for f in files(input_path):
        data = reader.read(f)
        data = converter.convert(data=data, **params)
        path = output_path.joinpath(f.stem).with_suffix(suffix)
        data = writer.write(path, data)


def read(path: str) -> List[Any] | Any:
    """read data from file or dir"""

    path: Path = Path(path)

    if path.is_file():
        return reader.read(path)

    data = [reader.read(f) for f in files(path)]
    return list(filter(lambda d: d is not None, data))


def write(path: str, data: Any) -> None:
    """writes data to file"""

    return writer.write(path, data)
