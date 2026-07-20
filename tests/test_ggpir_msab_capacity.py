"""Tests for the configurable capacity of the GGPIR multi-scale encoder/decoder.

Importing ``flows.core`` first mirrors the application import order and avoids a
latent circular import that only triggers when ``flows.ml.models`` is imported
before ``flows.core``.
"""
import flows.core  # noqa: F401  (ordering side effect, see module docstring)

import pytest
import torch

from flows.ml.models.ggpir import GGPIRMSABEncoder, GGPIRMSABDecoder


def _num_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.unit
def test_encoder_capacity_preserves_output_shape():
    x = torch.randn(2, 31, 16, 16)
    base = GGPIRMSABEncoder(in_channels=31, out_channels=3, capacity=1)
    wide = GGPIRMSABEncoder(in_channels=31, out_channels=3, capacity=3)

    assert base(x).shape == wide(x).shape
    # Higher capacity must yield a strictly larger model.
    assert _num_params(wide) > _num_params(base)


@pytest.mark.unit
def test_encoder_default_capacity_is_one():
    x = torch.randn(1, 31, 8, 8)
    default = GGPIRMSABEncoder(in_channels=31, out_channels=3)
    explicit = GGPIRMSABEncoder(in_channels=31, out_channels=3, capacity=1)
    assert _num_params(default) == _num_params(explicit)
    assert default(x).shape == explicit(x).shape


@pytest.mark.unit
def test_decoder_capacity_preserves_output_shape():
    encoder = GGPIRMSABEncoder(in_channels=31, out_channels=3, capacity=1)
    x = torch.randn(2, 31, 16, 16)
    latent = encoder(x)

    base = GGPIRMSABDecoder(in_channels=3, out_channels=31, capacity=1)
    wide = GGPIRMSABDecoder(in_channels=3, out_channels=31, capacity=2)

    out_base = base(latent)
    out_wide = wide(latent)

    assert out_base.shape == out_wide.shape == (2, 31, 16, 16)
    assert _num_params(wide) > _num_params(base)


@pytest.mark.unit
@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity_raises(capacity):
    with pytest.raises(ValueError):
        GGPIRMSABEncoder(in_channels=31, out_channels=3, capacity=capacity)
    with pytest.raises(ValueError):
        GGPIRMSABDecoder(in_channels=3, out_channels=31, capacity=capacity)
