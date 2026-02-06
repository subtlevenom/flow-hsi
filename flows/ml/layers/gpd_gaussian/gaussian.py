import torch
from einops import einsum

class Gaussian(torch.nn.Module):

    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, x:torch.Tensor, m:torch.Tensor, S:torch.Tensor):
        x = x - m
        xs = torch.einsum('bcij,bclij->blij', x, S)
        xsx = torch.einsum('bcij,bcij->bij', xs, x)

        s = torch.permute(S, (0, 3, 4, 1, 2))
        s = torch.linalg.det(s)
        s = torch.clip(s, 1e-7)

        p = torch.sqrt(s / (2 * torch.pi)**x.shape[1]) * torch.exp(-0.5 * xsx)
        return p.unsqueeze(1)
