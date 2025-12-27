import torch
from einops import einsum

class Gaussian(torch.nn.Module):

    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, x:torch.Tensor, m:torch.Tensor, S:torch.Tensor):
        x = x - m # BCHW
        # S = torch.linalg.inv(S) # BCSHW
        y = torch.einsum('bchw,bcshw->bshw', x, S)
        y = torch.sum(y * x, dim=1, keepdim=True)
        y = torch.exp(-0.5 * y)
        return y
