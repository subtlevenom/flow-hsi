import torch
from einops import einsum

class Gaussian(torch.nn.Module):

    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, x:torch.Tensor, c:torch.Tensor, a:torch.Tensor, m:torch.Tensor, s:torch.Tensor):
        return self.forward_mul(x, c, a, m, s)
        # return self.forward_sum(x, a, m, s)
        # return self.forward_poly(x, a, m, s)

    def forward_mul(self, x:torch.Tensor, c:torch.Tensor, a:torch.Tensor, m:torch.Tensor, s:torch.Tensor):
        x1 = torch.sum(c[:,:3] * x, dim=1, keepdim=True)
        x2 = torch.sum(c[:,3:] * x, dim=1, keepdim=True)
        x = torch.concat([x1,x2], dim=1)
        x = (x - m) # BCHW
        x = torch.sum(x * x, dim=1, keepdim=True)
        x = s + a * torch.exp(-0.5 * x)
        return x

    def forward_sum(self, x:torch.Tensor, a:torch.Tensor, m:torch.Tensor, s:torch.Tensor):
        x = (x - m) * s # BCHW
        x = a * torch.exp(-0.5 * x * x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x

    def forward_poly(self, x:torch.Tensor, a:torch.Tensor, m:torch.Tensor, s:torch.Tensor):
        x = a + m * x + s *x*x
        x = torch.sum(x, dim=1, keepdim=True)
        #x = a * (x - m) * (x - m)
        #x = torch.sum(x, dim=1, keepdim=True)
        return x
