import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim

from flows.core import Logger
from flows.ml.models.ggpir.spectral_encoder_decoder import SpectralEncoderDecoder
from . import usgs_unet


class SpectralExperimentPipeline(usgs_unet.USGSUNetPipeline):
    """Independent masked reconstruction or transferred-encoder restoration."""

    def __init__(
        self, model: nn.Module, phase: str = "restoration",
        encoder_checkpoint: Optional[str] = None,
        allow_random_encoder: bool = False,
        encoder_lr: float = 2e-5,
        freeze_encoder_steps: int = 500,
        visible_weight: float = 0.1,
        **loss_params,
    ):
        super().__init__(model, pretrain_epochs=0, **loss_params)
        if phase not in ("pretrain", "restoration"):
            raise ValueError("phase must be 'pretrain' or 'restoration'.")
        if encoder_lr <= 0 or freeze_encoder_steps < 0 or visible_weight < 0:
            raise ValueError("Require encoder_lr > 0, freeze_encoder_steps >= 0, visible_weight >= 0.")
        if phase == "pretrain" and encoder_checkpoint is not None:
            raise ValueError("Pretraining starts its own encoder; encoder_checkpoint must be null.")
        if self.network.head_type != phase:
            raise ValueError("The pipeline phase must match the model head.")
        self.phase = phase
        self.encoder_checkpoint = encoder_checkpoint
        self.allow_random_encoder = allow_random_encoder
        self.encoder_lr = encoder_lr
        self.freeze_encoder_steps = freeze_encoder_steps
        self.visible_weight = visible_weight
        self._resuming = False
        self._best_masked_mae = math.inf
        self.save_hyperparameters(ignore=["model"])
        del self.hparams["pretrain_epochs"]

    @property
    def network(self) -> SpectralEncoderDecoder:
        if isinstance(self.model, SpectralEncoderDecoder):
            return self.model
        # Keep Flow's existing src -> res API, without registering the same
        # network twice in the state dictionary.
        if hasattr(self.model, "layers") and "spectral" in self.model.layers:
            network = self.model.layers["spectral"]
            if isinstance(network, SpectralEncoderDecoder):
                return network
        raise ValueError("Expected SpectralEncoderDecoder or Flow with a 'spectral' layer.")

    @property
    def checkpoint_monitor(self) -> str:
        return "val_masked_mae" if self.phase == "pretrain" else "val_mrae"

    @property
    def pretraining(self) -> bool:
        return self.phase == "pretrain"

    @property
    def encoder_frozen(self) -> bool:
        # Global step counts optimizer updates, including gradient accumulation.
        # Epoch cap keeps a small dataset from spending many epochs head-only.
        return (not self.pretraining and self.current_epoch == 0
                and self.global_step < self.freeze_encoder_steps)

    def setup(self, stage: str) -> None:
        # Unlike the original pipeline, this experiment does not switch tasks
        # after a fixed epoch count.
        pass

    def forward(self, x, y=None, band_mask=None):
        return self.network(x, band_mask=band_mask)

    def load_encoder(self, path: str) -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(artifact, dict)
                or artifact.get("format") != "flow-hsi.spectral-encoder.v1"
                or artifact.get("architecture") != self.network.encoder.architecture()):
            raise ValueError(f"Incompatible encoder artifact: {path}.")
        self.network.encoder.load_state_dict(artifact["encoder_state_dict"], strict=True)
        Logger.info(f"Loaded pretrained spectral encoder from {path}.")

    def on_fit_start(self) -> None:
        if not self.pretraining and not self._resuming:
            if self.encoder_checkpoint is not None:
                self.load_encoder(self.encoder_checkpoint)
            elif not self.allow_random_encoder:
                raise ValueError(
                    "Restoration requires encoder_checkpoint from pretraining. "
                    "Set allow_random_encoder=true only for a scratch-training ablation."
                )
            else:
                Logger.info("Training restoration with an explicitly requested random encoder.")
        self._set_encoder_state()

    def _set_encoder_state(self) -> None:
        trainable = not self.encoder_frozen
        self.network.encoder.requires_grad_(trainable)
        self.network.encoder.train(self.training and trainable)

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self._set_encoder_state()
        self.log("encoder_frozen", float(self.encoder_frozen), on_step=True, prog_bar=True)

    def reconstruction_losses(self, pred, target, mask):
        hidden = self.masked_mae(pred, target, mask)
        visible = self.masked_mae(pred, target, ~mask)
        structural = self.ssim_loss(pred, target) if self.ssim_weight else pred.new_zeros(())
        return {"loss": hidden + self.visible_weight * visible + self.ssim_weight * structural,
                "masked_mae": hidden, "visible_mae": visible, "ssim_loss": structural}

    def training_step(self, batch, batch_idx):
        if not self.pretraining:
            return super().training_step(batch, batch_idx)
        source, target = batch
        # Flip the starting domain each epoch, including single-batch datasets.
        reconstruction = source if (batch_idx + self.current_epoch) % 2 == 0 else target
        masked, mask = self.mask_channels(reconstruction)
        pred = self(masked, band_mask=mask)
        terms = self.reconstruction_losses(pred, reconstruction, mask)
        for name in ("masked_mae", "visible_mae", "ssim_loss"):
            self.log(name, terms[name], on_step=True, on_epoch=True,
                     batch_size=reconstruction.shape[0])
        self.log("train_loss", terms["loss"], prog_bar=True, batch_size=reconstruction.shape[0])
        self.log_quality(pred, reconstruction, "train")
        return {"loss": terms["loss"]}

    def _evaluate(self, batch, prefix: str, batch_idx: int):
        if not self.pretraining:
            return super()._evaluate(batch, prefix, batch_idx)
        _, target = batch
        masked, mask = self.mask_channels(target, deterministic=True, seed=batch_idx)
        pred = self(masked, band_mask=mask)
        terms = self.reconstruction_losses(pred, target, mask)
        for name, value in terms.items():
            self.log(f"{prefix}_{name}", value, on_step=False, on_epoch=True,
                     prog_bar=name in ("loss", "masked_mae"),
                     batch_size=target.shape[0], sync_dist=True)
        self.log_quality(pred, target, prefix)
        result = {"loss": terms["loss"]}
        if batch_idx == 0:
            result.update(image=pred.detach(), preview_source=masked.detach())
        return result

    def predict_step(self, batch, batch_idx):
        if self.pretraining:
            raise ValueError("Use pretraining test/validation for masked reconstruction, not predict.")
        return super().predict_step(batch, batch_idx)

    def on_validation_end(self) -> None:
        if not self.pretraining or self.trainer.sanity_checking:
            return
        metric = self.trainer.callback_metrics.get("val_masked_mae")
        if metric is None:
            return
        value = float(metric)
        if not math.isfinite(value):
            raise ValueError("Cannot export an encoder with nonfinite validation masked MAE.")
        if value > self._best_masked_mae:
            return
        self._best_masked_mae = value
        if self.trainer.is_global_zero:
            directory = Path(self.trainer.log_dir or self.trainer.default_root_dir) / "checkpoints"
            directory.mkdir(parents=True, exist_ok=True)
            artifact = {
                "format": "flow-hsi.spectral-encoder.v1",
                "architecture": self.network.encoder.architecture(),
                "encoder_state_dict": {
                    key: tensor.detach().cpu()
                    for key, tensor in self.network.encoder.state_dict().items()
                },
                "epoch": self.current_epoch, "global_step": self.global_step,
                "val_masked_mae": value,
            }
            temporary = directory / "encoder.pt.tmp"
            torch.save(artifact, temporary)
            os.replace(temporary, directory / "encoder.pt")

    def on_save_checkpoint(self, checkpoint) -> None:
        metric = self.trainer.callback_metrics.get("val_masked_mae") if self.pretraining else None
        checkpoint["spectral_experiment"] = {
            "phase": self.phase, "architecture": self.network.encoder.architecture(),
            "best_masked_mae": min(self._best_masked_mae, float(metric))
            if metric is not None else self._best_masked_mae,
            "freeze_encoder_steps": self.freeze_encoder_steps,
        }

    def on_load_checkpoint(self, checkpoint) -> None:
        metadata = checkpoint.get("spectral_experiment", {})
        if (metadata.get("phase") != self.phase
                or metadata.get("architecture") != self.network.encoder.architecture()
                or metadata.get("freeze_encoder_steps") != self.freeze_encoder_steps):
            raise ValueError("Resume requires matching spectral phase, encoder, and freeze schedule.")
        self._best_masked_mae = metadata["best_masked_mae"]
        self._resuming = True

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {"params": self.network.encoder.parameters(), "lr": self.encoder_lr
             if not self.pretraining else self.lr},
            {"params": self.network.decoder.parameters(), "lr": self.lr},
        ], weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=0,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
