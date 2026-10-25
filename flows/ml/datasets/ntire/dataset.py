import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2 import Compose, ToDtype, ToImage


def load_cube(path: str) -> np.memmap:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot memory-map NTIRE cube {path}: {error}") from error


class Dataset(TorchDataset):
    """Paired HWC NPY cubes, cropped before any full-array copy or conversion."""

    def __init__(
        self, paths_a: List[str], paths_b: List[str],
        transform: Optional[Compose] = None, filename: bool = False,
        norm: float = 1.0, crop_size: int = 0, random_crop: bool = False,
        flip_probability: float = 0.0, spectral_channels: int = 31,
    ) -> None:
        if not paths_a or len(paths_a) != len(paths_b):
            raise ValueError("Source and target must contain the same nonzero number of cubes.")
        if any(Path(a).name != Path(b).name for a, b in zip(paths_a, paths_b)):
            raise ValueError("Source and target filenames must match for every pair.")
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("norm must be finite and positive.")
        if crop_size < 0 or (crop_size and crop_size < 4):
            raise ValueError("crop_size must be zero or at least four pixels.")
        if not 0 <= flip_probability <= 1:
            raise ValueError("flip_probability must lie in [0, 1].")
        self.paths_a = paths_a
        self.paths_b = paths_b
        self.transform = transform if transform is not None else Compose([
            ToImage(), ToDtype(dtype=torch.float32, scale=True),
        ])
        self.filename = filename
        self.norm = norm
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.flip_probability = flip_probability
        self.spectral_channels = spectral_channels

    def __getitem__(self, idx: int):
        source_path, target_path = self.paths_a[idx], self.paths_b[idx]
        # Keep maps local to this call: workers pickle filenames, never open maps.
        # Errors propagate with the filename instead of the generic reader's None.
        source = load_cube(source_path)
        target = load_cube(target_path)
        if (source.ndim != 3 or source.shape != target.shape
                or source.shape[-1] != self.spectral_channels
                or min(source.shape[:2]) < 4):
            raise ValueError(
                f"Expected matching HWC cubes with {self.spectral_channels} bands and "
                f"H,W >= 4: {source_path} {source.shape}, {target_path} {target.shape}."
            )
        if source.dtype.kind not in "fui" or target.dtype.kind not in "fui":
            raise ValueError(f"Cubes must have real numeric dtypes: {source_path}, {target_path}.")

        h, w = source.shape[:2]
        top = left = 0
        if self.crop_size:
            if self.crop_size > min(h, w):
                raise ValueError(f"crop_size={self.crop_size} exceeds cube size in {source_path}.")
            if self.random_crop:
                top = int(torch.randint(h - self.crop_size + 1, (1,)).item())
                left = int(torch.randint(w - self.crop_size + 1, (1,)).item())
            else:
                top, left = (h - self.crop_size) // 2, (w - self.crop_size) // 2
            h = w = self.crop_size
        # Preserve the reader's existing multiple-of-four output convention.
        h, w = h - h % 4, w - w % 4
        source = source[top:top + h, left:left + w]
        target = target[top:top + h, left:left + w]
        for axis in (1, 0):
            if self.flip_probability and torch.rand(()).item() < self.flip_probability:
                source, target = np.flip(source, axis), np.flip(target, axis)

        # Own writable CHW storage for just the selected patch. ToImage accepts
        # tensors without making its usual full-cube HWC -> contiguous CHW copy.
        x = torch.from_numpy(np.array(source.transpose(2, 0, 1), copy=True, order="C"))
        y = torch.from_numpy(np.array(target.transpose(2, 0, 1), copy=True, order="C"))
        x, y = self.transform(x), self.transform(y)
        if self.norm != 1.0:
            x = x / self.norm
        if self.filename:
            return x, y, Path(target_path).name
        return x, y

    def __len__(self) -> int:
        return len(self.paths_a)
