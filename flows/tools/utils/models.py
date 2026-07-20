import torch
from torch import nn


def load_model(model: nn.Module, key: str, path: str, strict: bool = True):
    """Loads a sub-module state dict from a checkpoint.

    Weights are matched by the ``key`` prefix (e.g. ``model.layers.encoder``)
    and remapped to be relative to ``model`` before loading. Pass an empty
    ``key`` to load the whole checkpoint state dict as-is.
    """
    checkpoint: dict = torch.load(path, weights_only=False)['state_dict']
    weights_dict = {}
    model_key = f'{key}.' if key else ''
    for ckpt_key, value in checkpoint.items():
        if ckpt_key.startswith(model_key):
            weights_dict[ckpt_key[len(model_key):]] = value

    if not weights_dict:
        raise KeyError(
            f'No weights matching key "{key}" were found in checkpoint "{path}"')

    return model.load_state_dict(weights_dict, strict=strict)


def require_grad(model: nn.Module, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
    model.requires_grad_ = requires_grad
