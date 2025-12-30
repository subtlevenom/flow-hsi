import os
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from rich.progress import Progress
import cv2
import random
import numpy as np
import imageio
import albumentations as A
import shutil
from omegaconf import DictConfig
from tools.utils.concurrent import concurrent
from tools.utils import images
from tools.files import reader
from scipy import io
from tools.files.iterators import files

THREADS = 1

SOURCE = 'source'
TARGET = 'target'

TEST = {
    'oil_painting.npy',
    'paints.npy',
    'photo_and_face.npy',
    'pompoms.npy',
    'real_and_fake_apples.npy',
    'real_and_fake_peppers.npy',
    'sponges.npy',
    'stuffed_toys.npy',
    'superballs.npy',
    'thread_spools.npy',
    'watercolors.npy',
}


def split(source_path:str, target_path:str, output_path:str, split:dict) -> None:
    """
    CAVE-HSI 
    """

    source_dir = Path(source_path)
    target_dir = Path(target_path)
    output_dir = Path(output_path)

    if not source_dir.is_dir():
        raise Exception(f'No such directory: {source_dir}')
    if not target_dir.is_dir():
        raise Exception(f'No such directory: {source_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)

    # test
    test_output_dir = output_dir.joinpath('test')
    test_output_dir.mkdir(parents=True, exist_ok=True)
    test_src_output_dir = test_output_dir.joinpath(SOURCE)
    test_src_output_dir.mkdir(parents=True, exist_ok=True)
    test_tgt_output_dir = test_output_dir.joinpath(TARGET)
    test_tgt_output_dir.mkdir(parents=True, exist_ok=True)
    for filename in TEST:
        try:
            # common files
            in_src_file = source_dir.joinpath(filename)
            in_tgt_file = target_dir.joinpath(filename)
            if not in_src_file.exists() or not in_tgt_file.exists():
                continue
            # source
            out_file = test_src_output_dir.joinpath(filename)
            shutil.copy(in_src_file, out_file)
            # target
            out_file = test_tgt_output_dir.joinpath(filename)
            shutil.copy(in_tgt_file, out_file)
        except Exception as e:
            print(f'Skip test file {filename} with exception {e}')
    
    source_files = set([f.name for f in files(source_dir, '.npy')]) - TEST
    target_files = set([f.name for f in files(target_dir, '.npy')]) - TEST
    common_files = list(source_files & target_files)

    # split
    random.shuffle(common_files)
    split_slices = np.cumsum([int(s * len(common_files)) for s in split.values()])
    split_files = np.split(common_files, split_slices[:-1])

    with Progress() as progress:
        for name, sfiles in zip(split.keys(), split_files):
            pb = progress.add_task(f"[cyan]{name}", total=2*len(sfiles))
            try:
                # source
                output_split_dir = output_dir.joinpath(name, SOURCE)
                output_split_dir.mkdir(parents=True, exist_ok=True)
                for file in sfiles:
                    input_path = source_dir.joinpath(file)
                    output_path = output_split_dir.joinpath(file)
                    shutil.copy(input_path, output_path)
                    progress.update(pb, advance=1)
                # target
                output_split_dir = output_dir.joinpath(name, TARGET)
                output_split_dir.mkdir(parents=True, exist_ok=True)
                for file in sfiles:
                    input_path = target_dir.joinpath(file)
                    output_path = output_split_dir.joinpath(file)
                    shutil.copy(input_path, output_path)
                    progress.update(pb, advance=1)

            except Exception as e:
                print(f'Skip split {name} with exception {e}')
