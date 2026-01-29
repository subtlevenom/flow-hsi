import torch
from torch import nn
import numpy as np

def covariance_matrix(a:torch.Tensor):
    n = int(np.sqrt(a.shape[1]))
    a = torch.unflatten(a, dim=1, sizes=(n, n))
    a = a + a.permute((0,2,1,3,4))
    return a


