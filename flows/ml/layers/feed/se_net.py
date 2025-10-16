from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SENet(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """

    def __init__(self,
                 n_features: int,
                 reduction: int = 4,
                 activation: int = nn.ReLU):
        super(SENet, self).__init__()

        if n_features % reduction != 0:
            raise ValueError(
                f'Number of features {n_features} must be divisible by reduction {reduction}'
            )

        activation = activation or nn.Sequential

        self.squeeze = nn.AdaptiveAvgPool1d(output_size=1)
        self.linear1 = nn.Conv1d(in_channels=n_features,
                                 out_channels=n_features // reduction,
                                 kernel_size=1)
        self.act1 = activation(inplace=True)
        self.linear2 = nn.Conv1d(in_channels=n_features // reduction,
                                 out_channels=n_features,
                                 kernel_size=1)
        self.act2 = activation(inplace=True)

    def forward(self, x: torch.Tensor):
        y = x.flatten(start_dim=2)
        y = self.squeeze(y)
        y = self.act1(self.linear1(y))
        y = self.act2(self.linear2(y))
        y = x * y.view(x.shape)
        return y
