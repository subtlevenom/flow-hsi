from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .se_net import SENet


class ResSENet(nn.Module):

    def __init__(self,
                 n_features: int,
                 reduction: int = 4,
                 activation: int = nn.ReLU):
        super(ResSENet, self).__init__()

        # convolutions

        self.norm1 = nn.BatchNorm2d(n_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=n_features,
                               out_channels=n_features,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.norm2 = nn.BatchNorm2d(n_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=n_features,
                               out_channels=n_features,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        # squeeze and excitation

        self.senet = SENet(n_features=n_features,
                           reduction=reduction,
                           activation=activation)

    def forward(self, x):

        # convolutions

        y = self.conv1(self.relu1(self.norm1(x)))
        y = self.conv2(self.relu2(self.norm2(y)))

        # squeeze and excitation

        y = self.senet(y)

        # add residuals

        y = torch.add(x, y)

        return y
