import torch.nn.functional as F

from flows.tools.utils.colors import rgb_to_lab
from . import shared_cmkan
from flows.ml.metrics.delta_e_2000 import _delta_E_CIE2000


class SprecherRGBPipeline(shared_cmkan.SharedCmKANPipeline):
    """Paired RGB restoration, without hyperspectral losses or weight reinitialization."""

    checkpoint_monitor = "val_loss"

    def __init__(self, model, lr=2e-4, weight_decay=1e-5, ssim_weight=0.15):
        super().__init__(
            model, spectral_channels=3, use_ntire_colorimetry=False,
            perceptual_weight=0.0, mrae_weight=0.0, spectral_weight=0.0,
            spectral_gradient_weight=0.0, lr=lr, weight_decay=weight_decay,
            ssim_weight=ssim_weight,
        )

    def restoration_losses(self, pred, target):
        if pred.shape != target.shape or pred.shape[1] != 3:
            raise ValueError("RGB restoration requires matching three-channel BCHW images.")
        pred, target = pred.float(), target.float()
        mae = F.l1_loss(pred, target)
        structural = self.ssim_loss(pred, target)
        return {"loss": mae + self.ssim_weight * structural,
                "mae": mae, "ssim_loss": structural}

    def quality_metrics(self, pred, target):
        pred, target = pred.float(), target.float()
        metrics = self.spectral_quality_metrics(pred, target)
        # RGB angle is not a hyperspectral SAM score.
        metrics.pop("sam")
        metrics["de"] = _delta_E_CIE2000(
            rgb_to_lab(pred.clamp(0, 1)), rgb_to_lab(target.clamp(0, 1)),
        ).mean()
        return metrics
