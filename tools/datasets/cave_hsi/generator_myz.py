from pathlib import Path
import numpy as np
from omegaconf import DictConfig
from tools.optics import create_cdf
from tools.files import reader, writer


def generate(input_path: str, params: DictConfig, optics: DictConfig, **kwargs) -> None:
    """
    CAVE-HSI: https://ieee-dataport.org/documents/cave-hsi
    """

    input_dir = Path(input_path)

    if not input_dir.is_dir():
        raise Exception(f'No such directory: {input_dir}')

    cdf = create_cdf(optics)

    for gt_path in Path(input_dir).glob('**/*.mat'):
        try:
            src_path = gt_path.parent.parent.joinpath(params.tag).joinpath(gt_path.name)
            src_image = cdf(gt_path, padding=params.padding)
            writer.write(src_path, src_image)
        except Exception as e:
            print(f'Failed to convert {gt_path}: {e}.')
