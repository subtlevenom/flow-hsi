from pathlib import Path
from typing import Any, List
from omegaconf import DictConfig
import numpy as np
from .iterators import files
from . import reader, writer, converter, comparer


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


def compare(
    src_path: str,
    ref_path: str,
    params: DictConfig,
    **kwargs,
) -> None:
    """
    src_path - input file or dir
    ref_path - reference file or dir
    params - converter dependent args.
        metrics: metrics name
        norm: normalization needed
    """

    src_path: Path = Path(src_path)
    ref_path: Path = Path(ref_path)

    def compare_file(src_file, ref_file, params):
        src = reader.read(src_file)
        ref = reader.read(ref_file)
        return comparer.compare(src=src, ref=ref, **params)

    if src_path.is_file():
        val = compare_file(src_path, ref_path, params)
        print(f'{params.metrics}: {val}')
    else:
        avg = 0
        n = 0
        for src_file in files(src_path):
            ref_file = ref_path.joinpath(src_file.name)
            val = compare_file(src_file, ref_file, params)
            print(f'{src_file.name}: {val}')
            avg += val
            n += 1
        print(f'{params.metrics}: {avg / n}')
