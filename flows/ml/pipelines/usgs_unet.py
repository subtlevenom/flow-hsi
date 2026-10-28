import math
from typing import Optional, Sequence

import lightning as L
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure

from flows.ml.layers.spectral_color import NTIREColorimetry
from flows.ml.metrics.delta_e_2000 import _delta_E_CIE2000


class USGSUNetPipeline(L.LightningModule):
    """Masked-channel pretraining followed by paired spectral restoration.

    Model constructors own initialization: in particular, Gaussian amplitudes
    and CM output heads must keep their identity-preserving zero initialization.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        pretrain_epochs: int = 5,
        mask_ratio: float = 0.5,
        mrae_weight: float = 0.1,
        spectral_weight: float = 0.1,
        spectral_gradient_weight: float = 0.1,
        color_weight: float = 0.0,
        spectral_response: Optional[Sequence[Sequence[float]]] = None,
        perceptual_weight: float = 0.1,
        masked_val_batches: int = 1,
        ssim_weight: float = 0.0,
    ):
        super().__init__()
        if pretrain_epochs < 0 or not 0.0 < mask_ratio < 1.0:
            raise ValueError("Require pretrain_epochs >= 0 and 0 < mask_ratio < 1.")
        if lr <= 0 or weight_decay < 0:
            raise ValueError("Require lr > 0 and weight_decay >= 0.")
        if masked_val_batches < 0:
            raise ValueError("masked_val_batches must be nonnegative.")
        if min(mrae_weight, spectral_weight, spectral_gradient_weight,
               color_weight, perceptual_weight, ssim_weight) < 0:
            raise ValueError("Loss weights must be nonnegative.")
        response = None
        if spectral_response is not None:
            response = torch.as_tensor(spectral_response, dtype=torch.float32)
            if (response.ndim != 2 or response.shape[0] != 3
                    or response.shape[1] < 2
                    or not torch.isfinite(response).all()
                    or (response < 0).any() or (response.sum(dim=1) <= 0).any()):
                raise ValueError("spectral_response must be a finite nonnegative [3, C] matrix.")
        if color_weight > 0 and response is None:
            raise ValueError("color_weight requires a calibrated spectral_response.")

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrain_epochs = pretrain_epochs
        self.mask_ratio = mask_ratio
        self.mrae_weight = mrae_weight
        self.spectral_weight = spectral_weight
        self.spectral_gradient_weight = spectral_gradient_weight
        self.color_weight = color_weight
        self.perceptual_weight = perceptual_weight
        self.masked_val_batches = masked_val_batches
        self.ssim_weight = ssim_weight
        self.colorimetry = NTIREColorimetry()
        self.register_buffer("spectral_response", response)
        self.save_hyperparameters(ignore=["model"])

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.pretrain_epochs >= self.trainer.max_epochs:
            raise ValueError("epochs must exceed pretrain_epochs to include paired fine-tuning.")

    @property
    def pretraining(self) -> bool:
        return self.current_epoch < self.pretrain_epochs

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        return self.model(src=x)["res"]

    def mask_channels(
        self, target: torch.Tensor, deterministic: bool = False, seed: int = 0,
    ):
        if target.ndim != 4 or target.shape[1] < 2:
            raise ValueError("Expected a hyperspectral [B, C, H, W] cube with C >= 2.")
        batch, channels = target.shape[:2]
        count = max(1, min(channels - 1, round(channels * self.mask_ratio)))
        # Local CPU generator makes validation masks repeatable without changing
        # the training RNG, including when validation runs on a different device.
        generator = torch.Generator().manual_seed(seed) if deterministic else None
        scores = torch.rand(batch, channels, generator=generator, device="cpu")
        indices = scores.argsort(dim=1)[:, :count].to(target.device)
        mask = torch.zeros(batch, channels, device=target.device, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        mask = mask[:, :, None, None]
        return target.masked_fill(mask, 0.0), mask

    @staticmethod
    def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        error = (pred.float() - target.float()).abs()
        return (error * mask).sum() / (mask.sum() * target.shape[-2] * target.shape[-1])

    @staticmethod
    def mrae(pred: torch.Tensor, target: torch.Tensor):
        return ((pred - target).abs() / target.abs().clamp_min(1e-3)).mean()

    @staticmethod
    def ssim_loss(pred: torch.Tensor, target: torch.Tensor):
        # Standard 11x11 Gaussian SSIM, shrinking only for very small patches.
        sigma = max(0.1, min(1.5, (min(pred.shape[-2:]) - 1) / 7))
        with torch.autocast(device_type=pred.device.type, enabled=False):
            # Older TorchMetrics crops away the Gaussian border even when
            # that leaves an empty image for tiny patches.
            padding = int(3.5 * sigma + 0.5)
            if min(pred.shape[-2:]) <= 2 * padding or padding == 0:
                _, image = structural_similarity_index_measure(
                    pred.float(), target.float(), data_range=1.0, sigma=sigma,
                    return_full_image=True,
                )
                return 1.0 - image.mean()
            return 1.0 - structural_similarity_index_measure(
                pred.float(), target.float(), data_range=1.0, sigma=sigma,
            )

    def restoration_losses(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.shape != target.shape:
            raise ValueError("Prediction and target cubes must have identical shapes.")
        pred, target = pred.float(), target.float()
        mae = F.l1_loss(pred, target)
        relative = self.mrae(pred, target)
        valid = target.norm(dim=1) > 1e-6
        cosine = F.cosine_similarity(pred, target, dim=1, eps=1e-6)
        spectral = ((1.0 - cosine).clamp_min(0) * valid).sum() / valid.sum().clamp_min(1)
        gradient = F.l1_loss(pred.diff(dim=1), target.diff(dim=1))
        color = pred.new_zeros(())
        if self.color_weight:
            if self.spectral_response.shape[1] != pred.shape[1]:
                raise ValueError("spectral_response must match the cube's channel count.")
            color = F.l1_loss(
                torch.einsum("rc,bchw->brhw", self.spectral_response, pred),
                torch.einsum("rc,bchw->brhw", self.spectral_response, target),
            )
        perceptual = pred.new_zeros(())
        if self.perceptual_weight:
            perceptual = F.l1_loss(self.colorimetry(pred), self.colorimetry(target)) / 100.0
        structural = self.ssim_loss(pred, target) if self.ssim_weight else pred.new_zeros(())
        loss = (mae + self.mrae_weight * relative + self.spectral_weight * spectral
                + self.spectral_gradient_weight * gradient + self.color_weight * color
                + self.perceptual_weight * perceptual + self.ssim_weight * structural)
        return {"loss": loss, "mae": mae, "mrae": relative,
                "spectral": spectral, "spectral_gradient": gradient, "color": color,
                "perceptual": perceptual, "ssim_loss": structural}

    @torch.no_grad()
    def quality_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        # PSNR uses all 31 bands and a fixed unit data range. DeltaE is CIEDE2000
        # in integrated Lab, not an arbitrary 3-band RGB slice.
        mse = (pred.float() - target.float()).square().flatten(1).mean(dim=1)
        psnr = (-10.0 * torch.log10(mse.clamp_min(1e-12))).mean()
        de = _delta_E_CIE2000(
            self.colorimetry(pred), self.colorimetry(target),
        ).mean().float()
        pred, target = pred.float(), target.float()
        pred_norm, target_norm = pred.norm(dim=1), target.norm(dim=1)
        cosine = (pred * target).sum(dim=1) / (
            pred_norm * target_norm
        ).clamp_min(torch.finfo(pred.dtype).tiny)
        # SAM is undefined at zero: identical zero spectra score 0, while
        # exactly one zero spectrum scores pi/2 rather than poisoning the mean.
        cosine = torch.where((pred_norm == 0) & (target_norm == 0), 1.0, cosine)
        sam = torch.rad2deg(cosine.clamp(-1.0, 1.0).acos()).mean()
        return {"psnr": psnr, "de": de, "sam": sam}

    def log_quality(self, pred, target, prefix):
        for name, value in self.quality_metrics(pred, target).items():
            self.log(f"{prefix}_{name}", value, prog_bar=True,
                     on_step=prefix == "train", on_epoch=True,
                     batch_size=target.shape[0], sync_dist=True)

    def training_step(self, batch, batch_idx):
        src, target = batch
        if self.pretraining:
            masked, mask = self.mask_channels(target)
            pred = self(masked)
            loss = self.masked_mae(pred, target, mask)
            self.log("masked_mae", loss, on_epoch=True, batch_size=target.shape[0])
        else:
            pred = self(src)
            terms = self.restoration_losses(pred, target)
            loss = terms["loss"]
            for name, value in terms.items():
                if name != "loss":
                    self.log(name, value, on_epoch=True, batch_size=target.shape[0])
        # During pretraining these describe masked reconstruction, not paired
        # restoration. Reuse its prediction rather than adding a model pass.
        self.log_quality(pred, target, "train")
        self.log("pretraining", float(self.pretraining), on_epoch=True,
                 batch_size=target.shape[0])
        self.log("train_loss", loss, prog_bar=True, batch_size=target.shape[0])
        return {"loss": loss}

    def _evaluate(self, batch, prefix: str, batch_idx: int):
        src, target = batch
        pred = self(src).float()
        terms = self.restoration_losses(pred, target)
        loss = terms["loss"]
        if prefix == "val" and self.pretraining and batch_idx < self.masked_val_batches:
            masked, mask = self.mask_channels(target, deterministic=True)
            masked_loss = self.masked_mae(self(masked), target, mask)
            self.log("val_masked_mae", masked_loss, batch_size=target.shape[0], sync_dist=True)
        # Always rank checkpoints on the same paired restoration task, even
        # during pretraining; never compare masked MAE against restoration MRAE.
        for name, value in terms.items():
            if name != "loss":
                self.log(f"{prefix}_{name}", value, prog_bar=name == "mrae",
                         batch_size=target.shape[0], sync_dist=True)
        self.log_quality(pred, target, prefix)
        self.log(f"{prefix}_loss", loss, batch_size=target.shape[0], sync_dist=True)
        result = {"loss": loss}
        if batch_idx == 0:
            result["image"] = pred.detach()
        return result

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self._evaluate(batch, "test", batch_idx)

    def predict_step(self, batch, batch_idx):
        src, _, names = batch
        return {"image": self(src), "filenames": names}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total = self.trainer.max_epochs

        def schedule(epoch: int):
            # Restart decay when switching tasks; Lightning restores last_epoch
            # from checkpoints so a resumed run stays in the correct phase.
            if epoch < self.pretrain_epochs:
                start, duration = 0, self.pretrain_epochs
            else:
                start, duration = self.pretrain_epochs, total - self.pretrain_epochs
            progress = min(1.0, (epoch - start) / max(1, duration))
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return {"optimizer": optimizer,
                "lr_scheduler": optim.lr_scheduler.LambdaLR(optimizer, schedule)}
