import torch
from torch import nn


def load_model(model: nn.Module, key:str, path:str):
    checkpoint: dict = torch.load(path, weights_only=False)['state_dict']
    weights_dict = {}
    model_key = f'{key}.'
    for key, value in checkpoint.items():
        if key.startswith(model_key):
            weights_dict[key[len(model_key):]] = value

    model.load_state_dict(weights_dict, strict=True)


def require_grad(model: nn.Module, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
    model.requires_grad_ = requires_grad
