"""Tests for pretrain/fine-tune transfer learning support.

Covers both the low-level ``flows.tools.utils.models`` helpers and the
``DefaultPipeline`` ``transfer`` configuration hook.

Importing ``flows.core`` first mirrors the application import order and avoids a
latent circular import that only triggers when ``flows.ml.models`` is imported
before ``flows.core``.
"""
import flows.core  # noqa: F401  (ordering side effect, see module docstring)

import pytest
import torch
from torch import nn

from flows.tools.utils import models
from flows.ml.pipelines.default import DefaultPipeline


class _TinyModel(nn.Module):
    """Minimal stand-in exposing a ``layers.encoder`` sub-module."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({
            'encoder': nn.Conv2d(3, 3, kernel_size=1),
            'decoder': nn.Conv2d(3, 3, kernel_size=1),
        })

    def forward(self, src=None, tgt=None):
        return {'res': src}


@pytest.mark.unit
def test_load_model_loads_matching_prefix(tmp_path):
    src = _TinyModel()
    for p in src.parameters():
        nn.init.constant_(p, 0.5)

    ckpt = {'state_dict': {f'model.{k}': v for k, v in src.state_dict().items()}}
    path = tmp_path / 'ckpt.ckpt'
    torch.save(ckpt, path)

    dst = _TinyModel()
    models.load_model(dst.layers.encoder, 'model.layers.encoder', str(path))

    assert torch.allclose(dst.layers.encoder.weight, src.layers.encoder.weight)
    # A different sub-module must remain untouched.
    assert not torch.allclose(dst.layers.decoder.weight, src.layers.decoder.weight)


@pytest.mark.unit
def test_load_model_missing_key_raises(tmp_path):
    src = _TinyModel()
    ckpt = {'state_dict': {f'model.{k}': v for k, v in src.state_dict().items()}}
    path = tmp_path / 'ckpt.ckpt'
    torch.save(ckpt, path)

    with pytest.raises(KeyError):
        models.load_model(_TinyModel().layers.encoder, 'does.not.exist', str(path))


@pytest.mark.unit
def test_require_grad_toggles_gradients():
    model = _TinyModel()
    models.require_grad(model.layers.encoder, False)
    assert all(not p.requires_grad for p in model.layers.encoder.parameters())
    assert all(p.requires_grad for p in model.layers.decoder.parameters())


@pytest.mark.unit
def test_pipeline_no_transfer_is_noop():
    pipeline = DefaultPipeline(model=_TinyModel())
    # Should not raise when transfer is unset.
    pipeline.setup('fit')


@pytest.mark.unit
def test_pipeline_transfer_loads_and_freezes(tmp_path):
    # Save a checkpoint from a fully-trained-looking pipeline.
    source = DefaultPipeline(model=_TinyModel())
    for p in source.model.parameters():
        nn.init.constant_(p, 0.25)
    path = tmp_path / 'pretrained.ckpt'
    torch.save({'state_dict': source.state_dict()}, path)

    transfer = [{
        'module': 'model.layers.encoder',
        'path': str(path),
        'freeze': True,
    }]
    target = DefaultPipeline(model=_TinyModel(), transfer=transfer)
    target.setup('fit')

    # Weights are transferred into the encoder ...
    assert torch.allclose(
        target.model.layers.encoder.weight,
        source.model.layers.encoder.weight,
    )
    # ... and the encoder is frozen while the rest of the model still trains.
    assert all(not p.requires_grad for p in target.model.layers.encoder.parameters())
    assert all(p.requires_grad for p in target.model.layers.decoder.parameters())


@pytest.mark.unit
def test_pipeline_transfer_missing_path_raises():
    target = DefaultPipeline(model=_TinyModel(), transfer=[{'module': 'model.layers.encoder'}])
    with pytest.raises(ValueError):
        target.setup('fit')
