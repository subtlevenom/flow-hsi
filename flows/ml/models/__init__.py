import sys, pkgutil, importlib, inspect
from torch import nn
import torch
from flows.ml import layers
from . import hskan, cmkan
from .flow import Flow


def create_layer(name: str, params) -> nn.Module:
    """creates module from known (imported) packages"""
    params = params or {}
    name = name.strip('. ')
    cls = sys.modules[__name__]
    for s in name.split('.'):
        cls = getattr(cls, s)
    return cls(**params)


def create_model(name: str, params: dict) -> Flow:
    names = [m.layer for m in params.metadata]
    modules = {
        n: create_layer(m.model, m.get('params', None))
        for n, m in params.layers.items() if n in names
    }
    return Flow(modules, params.metadata)
