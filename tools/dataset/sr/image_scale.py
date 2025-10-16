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

THREADS = 1

SOURCE = 'source'
TARGET = 'target'


def sample(config: DictConfig) -> None:
    """
    Target images sourced from the input folder, patched with crop_size
    Source images are targets scaled with bicubic
    """

    input_dir = Path(config.input)
    output_dir = Path(config.output)

    if not input_dir.is_dir():
        raise Exception(f'No such directory: {input_dir}')

    # all files in the source folder
    files = list(input_dir.glob('*.[jpg png bmp]*'))
    n_files = len(files)
    random.seed(config.seed)
    random.shuffle(files)

    # split files into buckets
    file_split = np.cumsum([int(p * n_files) for p in config.split.values()])
    file_split = np.split(files, file_split)

    splits = {}
    for name, files in zip(config.split.keys(), file_split):
        src_dir = output_dir.joinpath(name, SOURCE)
        tgt_dir = output_dir.joinpath(name, TARGET)
        src_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)
        splits[name] = {SOURCE: src_dir, TARGET: tgt_dir, 'files': files}

    tasks = []
    with Progress() as progress:
        for split_name, split_data in splits.items():
            pb = progress.add_task(f"[cyan]{split_name}",
                                   total=len(split_data['files']))
            with ThreadPoolExecutor(max_workers=THREADS) as executor:
                split_tasks = [
                    _prepare_data(executor,
                                  input_dir=input_dir,
                                  output_source_dir=split_data[SOURCE],
                                  output_target_dir=split_data[TARGET],
                                  filename=filename.name,
                                  params=config.params)
                    for filename in split_data['files']
                ]
                for task in split_tasks:
                    task.add_done_callback(
                        lambda _: progress.update(pb, advance=1))
                tasks.extend(split_tasks)

    _, not_done = wait(tasks, return_when=ALL_COMPLETED)

    if len(not_done) > 0:
        print(f'[Warn] Skipped {len(not_done)} image pairs.')


@concurrent
def _prepare_data(
    input_dir: Path,
    output_source_dir: Path,
    output_target_dir: Path,
    filename: str,
    params,
):
    input_path = input_dir.joinpath(filename)
    if not input_path.is_file():
        raise Exception('No source file')

    ref_size = params.crop_size
    src_size = int(params.crop_size / params.scale)
    random_crop = A.RandomCrop(height=ref_size, width=ref_size)

    image = imageio.v3.imread(input_path)
    for i in range(params.n_crops):
        save_name = Path(filename).stem + f'_{i}' + Path(filename).suffix
        try:
            # tgt
            img = random_crop(image=image)['image']
            imageio.v3.imwrite(output_target_dir.joinpath(save_name), img)
            # src
            img = cv2.resize(img, (src_size, src_size),
                             interpolation=cv2.INTER_CUBIC)
            imageio.v3.imwrite(output_source_dir.joinpath(save_name), img)
        except Exception as e:
            print(e)
