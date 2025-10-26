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
    val = psnr(torch.from_numpy(src), torch.from_numpy(ref)).numpy()
    return float(val)

