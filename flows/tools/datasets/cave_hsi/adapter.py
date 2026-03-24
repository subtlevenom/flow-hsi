import sys
from pathlib import Path
import numpy as np


def adapt(input_path: str, output_path: str, **kwargs) -> None:

    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    if not input_dir.is_dir():
        raise Exception(f'No such directory: {input_dir}')

    for dir in input_dir.iterdir():
        if not dir.is_dir():
            continue
        hsi = sorted(list(dir.glob('*/*.npy')))
        hsi = [np.load(s)[:,:,0] for s in hsi]
        hsi = np.stack(hsi, axis=2)

        output_path:Path = output_dir.joinpath(dir.name).with_suffix('.npy')
        np.save(output_path, hsi)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    adapt(input_dir, output_dir)