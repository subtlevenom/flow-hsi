from typing import Any, Callable, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pool(torch.nn.Module):
    """ Input features BxCxN """

    def forward(self, x: torch.Tensor):
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        return x
