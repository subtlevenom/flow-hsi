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

    sm = torch.zeros((B,H,W,C,C),dtype=a.dtype, device=a.device)
    for i in range(C):
        sm[:,:,:,i,i] = s[:,i]

    rs = torch.matmul(r,sm)
    r = torch.matmul(rs,r.permute(0,1,2,4,3))
    r = r.permute(0,3,4,1,2)
    return r


