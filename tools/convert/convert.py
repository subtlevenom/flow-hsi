from pathlib import Path
from typing import List, Tuple
import cv2 as cv
import numpy as np
import colour
from scipy.signal.windows import gaussian
# np.set_printoptions(threshold=sys.maxsize)
from tools.utils.images import read
from scipy import io
from tools.utils.hsi import hs_to_rgb


def convert(input_path: str, output_path: str):
    """
    input_path: input hs image file or dir path
    output: output file or dir path
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        convert_dir(input_path, output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_file(input_path, output_path)


def convert_file(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f'File {input_path} does not exist')
        return

    try:
        hs_image = io.loadmat(input_path)['hsi']
        rgb_image = hs_to_rgb(hs_image, CHANNELS)

    except Exception as e:
        print(f'Failed converting file {input_path} with exception {e}.')


def convert_dir(input_path: Path, output_path: Path):

    for file in input_path.glob('*.mat'):
        hs_image = io.loadmat(file)['hsi']
        rgb_image = hs_to_rgb(hs_image, )
