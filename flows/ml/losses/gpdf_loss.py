import torch
from torch import nn
from torchmetrics import Metric
import torch.nn.functional as F
import numpy as np


class GPDFLoss(Metric):

    def __init__(self):
        super(GPDFLoss, self).__init__()
        self.add_state("correct",
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("total",
                       default=torch.tensor(0, dtype=torch.long),
                       dist_reduce_fx="sum")

    def update(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        S: torch.Tensor,
    ) -> None:
        x = x - m
        xs = torch.einsum('bcij,bclij->blij', x, S)
        xsx = torch.einsum('bcij,bcij->bij', xs, x)

        s = torch.permute(S, (0, 3, 4, 1, 2))
        s = torch.linalg.det(s)
        s = torch.clip(s, 1e-7)

        loss = torch.mean(x.shape[1] * np.log(2 * np.pi) - torch.log(s) + xsx)

        self.correct += loss
        self.total += 1

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total
