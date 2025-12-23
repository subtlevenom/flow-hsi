import sys
from pathlib import Path
import numpy as np
from tools.files import reader
from tools.files.iterators import files


def normalize(gt_path: str, src_path: str, output_path: str, **kwargs) -> None:

    gt_dir = Path(gt_path)
    src_dir = Path(src_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    if not gt_dir.is_dir():
        raise Exception(f'No such directory: {gt_dir}')
    if not src_dir.is_dir():
        raise Exception(f'No such directory: {src_dir}')

    gt, src = [],[]
    for gt_file in files(gt_dir, '.npy'):
        src_file = src_dir.joinpath(gt_file.name)

        gt_img = reader.read(gt_file)
        if gt_img is None:
            continue

        src_img = reader.read(src_file)
        if src_img is None:
            continue

        gt.append(gt_img.reshape(-1, gt_img.shape[-1]))
        src.append(src_img.reshape(-1, src_img.shape[-1]))

    gt = np.concat(gt, axis=0) # n x 31
    src = np.concat(src, axis=0) # n x 31

    m,c = [],[]
    for n in range(src.shape[-1]):
        x = src[...,n]
        y = gt[...,n]
        _m, _c = np.polyfit(x, y, 1)
        m.append(_m)
        c.append(_c)
    norm = (np.array(m), np.array(c))

    for src_file in files(src_dir, '.npy'):

        src_image = reader.read(src_file)
        src_image = norm[0] * src_image + norm[1]

        gt_img = reader.read(gt_file)
        src_img = reader.read(src_file)

        output_file = output_dir.joinpath(src_file.name)
        np.save(output_file, src_img)

if __name__ == '__main__':
    gt_dir = sys.argv[1]
    src_dir = sys.argv[2]
    output_dir = sys.argv[3]
    normalize(gt_dir, src_dir, output_dir)