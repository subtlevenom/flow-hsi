from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from rich.progress import Progress
import cv2
import random
import numpy as np
import imageio
import albumentations as A
from omegaconf import DictConfig
from tools.utils.concurrent import concurrent
from tools.utils import images
from tools.files import reader
from scipy import io

THREADS = 1

SOURCE = 'source'
TARGET = 'target'


def sample(input_path:str, output_path:str, split:dict, params:DictConfig) -> None:
    """
    CAVE-HSI: https://ieee-dataport.org/documents/cave-hsi
    """

    input_dir = Path(input_path)
    output_dir = Path(output_path)

    if not input_dir.is_dir():
        raise Exception(f'No such directory: {input_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_filenames = []
    norm_src_files = {}
    norm_tgt_files = {}

    # Copy pairs (output_dir, input_path) splitted by tasks
    splits = {}
    for name, data in split.items():
        output_src_dir = output_dir.joinpath(name, SOURCE)
        output_src_dir.mkdir(parents=True, exist_ok=True)
        input_src_files = list(Path(data.source).glob('*.npy'))

        output_tgt_dir = output_dir.joinpath(name, TARGET)
        output_tgt_dir.mkdir(parents=True, exist_ok=True)
        input_tgt_files = list(Path(data.target).glob('*.mat'))

        # filter out unique names
        filenames = set([f.stem for f in input_src_files]) & set(
            [f.stem for f in input_tgt_files])

        input_src_files = {f.stem: f for f in input_src_files}
        input_tgt_files = {f.stem: f for f in input_tgt_files}

        src_files = [(output_src_dir, input_src_files[f]) for f in filenames]
        tgt_files = [(output_tgt_dir, input_tgt_files[f]) for f in filenames]

        splits[name] = list(zip(src_files, tgt_files))

        # norm
        norm_filenames.extend(filenames)
        norm_src_files = norm_src_files | input_src_files
        norm_tgt_files = norm_tgt_files | input_tgt_files

    # normalization
    norm = {}
    for filename in norm_filenames:
        src_file = norm_src_files[filename]
        tgt_file = norm_tgt_files[filename]
        src_img = reader.read(src_file)
        tgt_img = reader.read(tgt_file)
        m, c = [], []
        for n in range(src_img.shape[-1]):
            x = src_img[:,:,n].ravel()
            y = tgt_img[:,:,n].ravel()
            _m, _c = np.polyfit(x, y, 1)
            m.append(_m)
            c.append(_c)
        norm[filename] = (np.array(m),np.array(c))

    # Copy concurrent
    tasks = []
    with Progress() as progress:
        for split_name, split_data in splits.items():
            pb = progress.add_task(f"[cyan]{split_name}",
                                   total=len(split_data))
            with ThreadPoolExecutor(max_workers=THREADS) as executor:
                split_tasks = [
                    _copy_data(executor,
                               output_src_dir=copy[0][0],
                               input_src_file=copy[0][1],
                               output_ref_dir=copy[1][0],
                               input_ref_file=copy[1][1],
                               norm=norm,
                               params=params) for copy in split_data
                ]
                for task in split_tasks:
                    task.add_done_callback(
                        lambda _: progress.update(pb, advance=1))
                tasks.extend(split_tasks)

    _, not_done = wait(tasks, return_when=ALL_COMPLETED)

    if len(not_done) > 0:
        print(f'[Warn] Skipped {len(not_done)} image pairs.')


@concurrent
def _copy_data(
    output_src_dir: Path,
    input_src_file: Path,
    output_ref_dir: Path,
    input_ref_file: Path,
    norm: tuple,
    params,
):
    if not input_src_file.is_file():
        raise Exception('No source file')
    if not input_ref_file.is_file():
        raise Exception('No source file')

    crop_size = params.get('crop_size', None)
    random_crop = None if crop_size is None else A.RandomCrop(height=crop_size,
                                                              width=crop_size)
    n_crops = 1 if random_crop is None else params.get('n_crops', 1)

    src_image = reader.read(input_src_file)
    ref_image = reader.read(input_ref_file)

    # normalize by channels (a white point)

    src_image = norm[input_src_file.stem][0] * src_image +  norm[input_src_file.stem][1]

    for i in range(n_crops):
        try:
            if random_crop:
                save_name = input_src_file.stem + f'_{i}'
                img = np.concat([src_image, ref_image], axis=-1)
                img = random_crop(image=img)['image']
                img = np.split(img, 2, axis=-1)
                src_img = img[0]
                tgt_img = img[1]
            else:
                save_name = input_src_file.stem
                src_img = src_image
                tgt_img = ref_image
            save_src_path = output_src_dir.joinpath(save_name)
            np.save(save_src_path, src_img)
            save_tgt_path = output_ref_dir.joinpath(save_name)
            np.save(save_tgt_path, tgt_img)
        except Exception as e:
            print(e)
