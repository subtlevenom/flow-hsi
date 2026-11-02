from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2 import Compose, ToDtype, ToImage
from torchvision.transforms.v2 import functional as F


class Dataset(TorchDataset):
    """Paired RGB images with aligned augmentation and deterministic evaluation."""

    def __init__(self, paths_a, paths_b, crop_size=0, random_crop=False,
                 flip_probability=0.0, filename=False):
        if not paths_a or len(paths_a) != len(paths_b):
            raise ValueError("Source and target must contain the same nonzero number of images.")
        if any(Path(a).name != Path(b).name for a, b in zip(paths_a, paths_b)):
            raise ValueError("Source and target filenames must match.")
        if crop_size < 0 or not 0 <= flip_probability <= 1:
            raise ValueError("Require crop_size >= 0 and flip_probability in [0, 1].")
        self.paths_a, self.paths_b = paths_a, paths_b
        self.crop_size, self.random_crop = crop_size, random_crop
        self.flip_probability, self.filename = flip_probability, filename
        self.transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])

    def _read(self, path):
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Cannot read Volga2K image {path}: {error}") from error

    def __getitem__(self, index):
        x, y = self._read(self.paths_a[index]), self._read(self.paths_b[index])
        if x.shape != y.shape:
            raise ValueError(f"Paired image shapes differ: {self.paths_a[index]}, "
                             f"{self.paths_b[index]}.")
        h, w = x.shape[-2:]
        if self.crop_size:
            size = self.crop_size
            if size > min(h, w):
                raise ValueError(f"crop_size={size} exceeds image size in {self.paths_a[index]}.")
            if self.random_crop:
                top = int(torch.randint(h - size + 1, ()).item())
                left = int(torch.randint(w - size + 1, ()).item())
            else:
                top, left = (h - size) // 2, (w - size) // 2
            x, y = F.crop(x, top, left, size, size), F.crop(y, top, left, size, size)
        if self.flip_probability and torch.rand(()).item() < self.flip_probability:
            x, y = F.hflip(x), F.hflip(y)
        if self.flip_probability and torch.rand(()).item() < self.flip_probability:
            x, y = F.vflip(x), F.vflip(y)
        if self.filename:
            return x, y, Path(self.paths_b[index]).name
        return x, y

    def __len__(self):
        return len(self.paths_a)
