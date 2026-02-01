import torch
from torch import nn
import numpy as np


def covariance_matrix(s:torch.Tensor, a:torch.Tensor):
    B,C,H,W = s.shape
    r = None
    cs = torch.cos(a)
    ss = torch.sin(a)
    c = 0
    for i in range(C-1):
        for j in range(i+1,C):
            rij = torch.zeros((B,H,W,C,C),dtype=a.dtype, device=a.device)
            rij[:,:,:] = torch.eye(6)
            cij = cs[:,c]
            sij = ss[:,c]
            rij[:,:,:,i,i] = cij
            rij[:,:,:,j,j] = cij
            rij[:,:,:,i,j] = -sij
            rij[:,:,:,j,i] = sij
            r = rij if c == 0 else torch.matmul(r, rij)
            c += 1

    # C
    CT = r
    C = r.permute(0,1,2,4,3)
    # D
    D = torch.zeros((B,H,W,C,C),dtype=a.dtype, device=a.device)
    D[:,:,:] = torch.eye(C)
    D = D * s.unsqueeze(2).permute(0,3,4,1,2)
    # CT*D*C
    S = torch.matmul(CT,D)
    S = torch.matmul(S,C)

    C = C.permute(0,3,4,1,2)
    D = D.permute(0,3,4,1,2)
    S = S.permute(0,3,4,1,2)

    return S,C,D


