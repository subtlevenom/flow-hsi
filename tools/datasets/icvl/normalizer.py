import sys
from pathlib import Path
import numpy as np
from tools.files import reader
from tools.files.iterators import files


def normalize(gt_path: str, src_path: str, output_path: str, **kwargs) -> None:

    gt_dir = Path(gt_path)
    src_dir = Path(src_path)
    output_dir = Path(output_path)

    if not gt_dir.is_dir():
        raise Exception(f'No such directory: {gt_dir}')
    if not src_dir.is_dir():
        raise Exception(f'No such directory: {src_dir}')

    # normalized src
    output_src_dir = output_dir.joinpath(src_dir.name)
    output_src_dir.mkdir(exist_ok=True, parents=True)
    # normalized gt
    output_gt_dir = output_dir.joinpath(gt_dir.name)
    output_gt_dir.mkdir(exist_ok=True, parents=True)

    gt_max = 0
    gt, src = [],[]
    for gt_file in files(gt_dir, '.npy'):
        src_file = src_dir.joinpath(gt_file.name)

        try:
            gt_img = reader.read(gt_file)
            if gt_img is None:
                continue
            gt_max = max(gt_max, np.max(gt_img))

            src_img = reader.read(src_file)
            if src_img is None:
                continue

            if gt_img.shape != src_img.shape:
                continue

            gt.append(gt_img.reshape(-1, gt_img.shape[-1])[::3])
            src.append(src_img.reshape(-1, src_img.shape[-1])[::3])
        except Exception as e:
            print(f'Skip {gt_file.name} with exception: {e}')

    gt = np.concat(gt, axis=0) / gt_max # n x 31
    src = np.concat(src, axis=0) # n x 31

    m,c = [],[]
    for n in range(src.shape[-1]):
        x = src[...,n]
        y = gt[...,n]
        _m, _c = np.polyfit(x, y, 1)
        m.append(_m)
        c.append(_c)
    norm = (np.array(m), np.array(c))

    # output gt
    for gt_file in files(gt_dir, '.npy'):

        gt_img = reader.read(gt_file)
        gt_img = gt_img / gt_max

        output_file = output_gt_dir.joinpath(gt_file.name)
        np.save(output_file, gt_img)

    # output src
    for src_file in files(src_dir, '.npy'):

        src_img = reader.read(src_file)
        src_img = norm[0] * src_img + norm[1]

        output_file = output_src_dir.joinpath(src_file.name)
        np.save(output_file, src_img)


if __name__ == '__main__':
    gt_dir = sys.argv[1]
    src_dir = sys.argv[2]
    output_dir = sys.argv[3]
    normalize(gt_dir, src_dir, output_dir)