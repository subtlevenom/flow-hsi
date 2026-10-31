from torch import optim

from . import usgs_unet


class GGPIRCascadePipeline(usgs_unet.USGSUNetPipeline):
    """Paired restoration only; no encoder transfer or masked reconstruction."""

    def __init__(self, model, min_lr=1e-6, **params):
        if params.get("pretrain_epochs", 0) != 0:
            raise ValueError("GGPIRCascade training requires pretrain_epochs=0.")
        params["pretrain_epochs"] = 0
        params["masked_val_batches"] = 0
        super().__init__(model, **params)
        if not 0 <= min_lr <= self.lr:
            raise ValueError("Require 0 <= min_lr <= lr.")
        self.min_lr = min_lr

    @property
    def checkpoint_monitor(self):
        return "val_psnr"

    @property
    def checkpoint_mode(self):
        return "max"

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.min_lr,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
