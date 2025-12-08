import math
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio as PSNR


def psnr(
        src: np.ndarray,
        ref: np.ndarray,
        data_range: tuple = (0, 1),
        reduction: str = 'elementwise_mean',
):
    psnr = PSNR(data_range=data_range, reduction=reduction)
    src = torch.from_numpy(src).permute(2, 0, 1).unsqueeze(0)
    ref = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0)
    val = psnr(src, ref).numpy()
    return float(val)

