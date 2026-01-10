import torch
from einops import einsum

class Gaussian(torch.nn.Module):

    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, x:torch.Tensor, a:torch.Tensor, m:torch.Tensor, s:torch.Tensor):
        return self.forward_mul(x, a, m, s)
        # return self.forward_sum(x, a, m, s)

    def forward_mul(self, x:torch.Tensor, a:torch.Tensor, m:torch.Tensor, s:torch.Tensor):
        x = (x - m) * s # BCHW
        x = torch.sum(x * x, dim=1, keepdim=True)
        x = a * torch.exp(-0.5 * x)
        return x

    def forward_sum(self, x:torch.Tensor, a:torch.Tensor, m:torch.Tensor, s:torch.Tensor):
        x = (x - m) * s # BCHW
        x = torch.exp(-0.5 * x * x)
        x = a * torch.sum(x, dim=1, keepdim=True)
        return x
