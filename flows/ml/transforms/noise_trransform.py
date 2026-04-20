import torch
from torch import nn
import random
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import RandomCrop
from torchvision.transforms import Compose
from typing import Tuple


class NoiseTransform(nn.Module):
    def __init__(self, sigma: float = 0.01, p: float = 0.5, seed: int = 42) -> None:
        super().__init__()
        self.p = p
        self.sigma = sigma
        random.seed(seed)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if random.random() < self.p and self.sigma > 0:
            # Генерируем шум той же размерности, что и image2
            noise = torch.randn_like(image) * self.sigma
            image = image + noise
            # Ограничиваем значения, чтобы не выйти за пределы [0, 1]
            image = torch.clamp(image, 0.0, 1.0)

        return image
