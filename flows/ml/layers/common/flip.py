import torch


class Flip(torch.nn.Module):

    def forward(self, x: torch.Tensor, dims=(-1, )) -> torch.Tensor:
        return torch.flip(x, dims=dims)
