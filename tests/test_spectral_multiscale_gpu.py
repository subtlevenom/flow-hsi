from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from flows.core import ModelSelector, PipelineSelector
from flows.ml.models.ggpir.spectral_encoder_decoder import SpectralEncoderDecoder


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
@pytest.mark.parametrize("mixed", [False, True])
def test_configured_gpu_batch_and_all_backbone_gradients(phase, mixed):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / ".configs"), version_base="1.3",
    ):
        cfg = compose(config_name=phase)
    module = PipelineSelector.select(ModelSelector.select(cfg.model), cfg.pipeline).cuda().train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=2e-4)
    target = torch.rand(8, 31, 196, 196, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
        if phase == "pretrain":
            masked, mask = module.mask_channels(target)
            prediction = module(masked, band_mask=mask)
            terms = module.reconstruction_losses(prediction, target, mask)
        else:
            prediction = module(target * 0.8)
            terms = module.restoration_losses(prediction, target)
    assert prediction.shape == target.shape
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in module.parameters())
    encoder = module.network.encoder
    for block in [*encoder.analysis, *encoder.synthesis, *encoder.skip_fusion,
                  encoder.detail, encoder.detail_fusion]:
        assert sum(p.grad.abs().sum() for p in block.parameters()) > 0
    optimizer.step()


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
def test_gpu_shapes_and_hidden_channel_masking(phase):
    model = SpectralEncoderDecoder(head=phase).cuda().eval()
    with torch.no_grad():
        for height, width in [(1, 1), (2, 3), (17, 19), (197, 199)]:
            x = torch.rand(1, 31, height, width, device="cuda")
            mask = torch.zeros(1, 31, 1, 1, dtype=torch.bool, device="cuda")
            mask[:, ::2] = True
            output = model(x, mask)
            assert output.shape == x.shape and torch.isfinite(output).all()
            torch.testing.assert_close(output, model(x.masked_fill(mask, 999), mask))
