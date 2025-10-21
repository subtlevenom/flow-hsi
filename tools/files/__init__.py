from pathlib import Path
from typing import Any, List
from omegaconf import DictConfig
from .iterators import files
from . import reader, writer, converter


def convert(
    input_path: str,
    output_path: str,
    params: DictConfig,
    **kwargs,
) -> None:
    """
    input_path - input file or dir
    output_path - output file or dir
    params - converter dependent args.
        converter: converter name
        format - output file suffix 
    """

    input_path: Path = Path(input_path)
    output_path: Path = Path(output_path)

    def convert_file(input_file, output_file, params):
        data = reader.read(input_file)
        data = converter.convert(data=data, **params)
        data = writer.write(output_file, data)

    if input_path.is_file():
        convert_file(input_path, output_path, params)
    else:
        for input_file in files(input_path):
            output_file = output_path.joinpath(input_file.stem).with_suffix(
                params.format)
            convert_file(input_file, output_file, params)


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
