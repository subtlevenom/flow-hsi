import sys
from pathlib import Path
import numpy as np
from scipy import io
import h5py

TAGS = ['cube', 'hsi', 'ref', 'rad']


def read(file: Path) -> np.ndarray:
    """reads hyperspectral image"""
    with h5py.File(file, 'r') as mat:
        for tag in TAGS: 
            data = mat.get(tag, None) 
            if data is not None:
                data = np.array(data)
                data = np.transpose(data) #.moveaxis(mat, 0, 2)
                return data

def convert(input_path: str, output_path: str, **kwargs) -> None:

    input_dir = Path(input_path)
    output_dir = Path(output_path)

    if not input_dir.is_dir():
        raise Exception(f'No such directory: {input_dir}')

    # normalized src
    output_dir.mkdir(exist_ok=True, parents=True)

    for f in input_dir.glob('**/*.mat'):
        output_file = output_dir.joinpath(f.name).with_suffix('.npy')

        try:
            img = read(f)
            if img is None:
                continue
            np.save(output_file, img)
        except Exception as e:
            print(f'Skip {f.name} with exception: {e}')

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    convert(input_dir, output_dir)