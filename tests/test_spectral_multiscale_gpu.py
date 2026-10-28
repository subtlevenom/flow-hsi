from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, TensorDataset

from flows.core import ModelSelector, PipelineSelector
from flows.ml.models.ggpir.spectral_encoder_decoder import SpectralEncoderDecoder
from flows.ml.pipelines.spectral import SpectralExperimentPipeline


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.fixture
def exact_cuda_math():
    cudnn = torch.backends.cudnn.allow_tf32
    matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32 = cudnn
    torch.backends.cuda.matmul.allow_tf32 = matmul


def configured(phase):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / ".configs"), version_base="1.3",
    ):
        cfg = compose(config_name=phase)
    return PipelineSelector.select(ModelSelector.select(cfg.model), cfg.pipeline)


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
@pytest.mark.parametrize("mixed", [False, True])
def test_configured_gpu_batch_and_all_scale_gradients(phase, mixed):
    module = configured(phase).cuda().train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=2e-4)
    target = torch.rand(8, 31, 196, 196, device="cuda")
    source = target * 0.8
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
        if phase == "pretrain":
            masked, mask = module.mask_channels(target)
            prediction = module(masked, band_mask=mask)
            terms = module.reconstruction_losses(prediction, target, mask)
        else:
            prediction = module(source)
            terms = module.restoration_losses(prediction, target)
    assert prediction.shape == target.shape
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in module.parameters())
    encoder = module.network.encoder
    connections = (
        encoder.small_to_medium, encoder.small_to_large, encoder.medium_to_large,
        encoder.small_to_extra_large, encoder.medium_to_extra_large,
        encoder.large_to_extra_large,
    )
    for connection in connections:
        assert connection.gate.weight.grad.abs().sum() > 0
        assert connection.project.weight.grad.abs().sum() > 0
    for branch in encoder.branches:
        assert branch[0].body[0].weight.grad.abs().sum() > 0
    optimizer.step()


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
def test_gpu_full_resolution_shapes_masking_and_execution_equivalence(phase, exact_cuda_math):
    model = SpectralEncoderDecoder(head=phase, checkpoint_blocks=False).cuda().eval()
    sizes = []
    handles = [
        path[-1].register_forward_hook(lambda m, args, out: sizes.append(tuple(out.shape)))
        for path in [*model.encoder.branches, model.encoder.detail]
    ]
    x = torch.rand(1, 31, 197, 199, device="cuda")
    with torch.no_grad():
        assert model(x).shape == x.shape
    assert sizes == [
        (1, 48, 197, 199), (1, 96, 197, 199), (1, 128, 197, 199),
        (1, 160, 197, 199), (1, 48, 197, 199),
    ]
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        for height, width in [(1, 1), (2, 3), (17, 19)]:
            tiny = torch.rand(1, 31, height, width, device="cuda")
            assert model(tiny).shape == tiny.shape
        full = torch.rand(1, 31, 256, 256, device="cuda")
        expected = model(full)
        patch = model(full[..., 27:223, 27:223])
        torch.testing.assert_close(
            patch[..., 64:132, 64:132], expected[..., 91:159, 91:159],
            atol=2e-5, rtol=2e-4,
        )
        mask = torch.zeros(1, 31, 1, 1, dtype=torch.bool, device="cuda")
        mask[:, ::2] = True
        changed = x.clone().masked_fill(mask, 999)
        torch.testing.assert_close(model(x, mask), model(changed, mask))
        first = model(x)
        model.encoder.pixel_chunk_size = 1000
        torch.testing.assert_close(model(x), first, atol=2e-5, rtol=2e-4)


def test_gpu_export_transfer_freeze_and_resume(tmp_path):
    def network(head):
        return SpectralEncoderDecoder(
            head=head, channels=8, branch_channels=(8, 12, 16, 20),
            branch_dilations=((1,), (1,), (1,), (1,)), heads=2,
        )

    def trainer(root, epochs=1):
        return L.Trainer(
            default_root_dir=root, accelerator="gpu", devices=1,
            max_epochs=epochs, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0,
        )

    source = torch.rand(2, 31, 16, 16)
    loader = DataLoader(TensorDataset(source, source * 0.8), batch_size=1)
    pretrain = SpectralExperimentPipeline(network("pretrain"), phase="pretrain")
    trainer(tmp_path / "pretrain").fit(pretrain, loader, loader)
    artifact = tmp_path / "pretrain" / "checkpoints" / "encoder.pt"
    assert artifact.is_file()
    finetune = SpectralExperimentPipeline(
        network("finetune"), phase="finetune",
        encoder_checkpoint=str(artifact), freeze_encoder_steps=1,
    )
    run = trainer(tmp_path / "finetune")
    run.fit(finetune, loader, loader)
    run.save_checkpoint(tmp_path / "full.ckpt")
    gate = finetune.network.encoder.large_to_extra_large.gate.weight
    assert run.optimizers[0].state[gate]["step"] == 1
    artifact.unlink()
    resumed = SpectralExperimentPipeline(
        network("finetune"), phase="finetune",
        encoder_checkpoint=str(artifact), freeze_encoder_steps=1,
    )
    trainer(tmp_path / "resume", epochs=2).fit(
        resumed, loader, loader, ckpt_path=tmp_path / "full.ckpt",
    )
    assert resumed._resuming
    assert resumed.global_step == 4
    with pytest.raises(ValueError, match="matching spectral encoder and decoder"):
        incompatible = SpectralExperimentPipeline(
            SpectralEncoderDecoder(head="finetune"), phase="finetune",
            freeze_encoder_steps=1,
        )
        incompatible.on_load_checkpoint(torch.load(tmp_path / "full.ckpt", weights_only=False))


def test_gpu_checkpointed_gradients_and_cross_scale_conditioning(exact_cuda_math):
    kwargs = dict(
        channels=8, branch_channels=(8, 12, 16, 20),
        branch_dilations=((1,), (1,), (1,), (1,)), heads=2,
    )
    checkpointed = SpectralEncoderDecoder(head="pretrain", **kwargs).cuda().train()
    plain = SpectralEncoderDecoder(
        head="pretrain", checkpoint_blocks=False, **kwargs,
    ).cuda().train()
    plain.load_state_dict(checkpointed.state_dict())
    x = torch.rand(2, 31, 17, 19, device="cuda")
    first, second = checkpointed(x), plain(x)
    torch.testing.assert_close(first, second)
    first.square().mean().backward()
    second.square().mean().backward()
    for p, q in zip(checkpointed.parameters(), plain.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
    connection = checkpointed.encoder.small_to_medium
    target = torch.rand(1, 12, 8, 8, device="cuda")
    donor = torch.rand(1, 8, 8, 8, device="cuda")
    assert not torch.allclose(connection(target, donor), connection(target, donor + 1))
    with torch.no_grad():
        connection.gate.weight.zero_()
        connection.gate.bias.fill_(-100)
        torch.testing.assert_close(connection(target, donor), target)
