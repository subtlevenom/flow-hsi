from typing import Any
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
import torch
import torchvision
from flows.tools.files import write


class GenerateCallback(Callback):
    def __init__(
            self,
            every_n_epochs=1,
            max_images=1,
            max_size=256,
        ) -> None:
        super().__init__()
        if every_n_epochs < 0 or max_images < 1 or max_size < 1:
            raise ValueError("Require every_n_epochs >= 0, max_images >= 1, max_size >= 1.")
        self.every_n_epochs = every_n_epochs
        self.max_images = max_images
        self.max_size = max_size
        self.save_dir = None

    @staticmethod
    def _to_rgb(imgs: torch.Tensor) -> torch.Tensor:
        '''Reduce a batch to 3 channels for previewing.

        torchvision/PIL can only save 1- or 3-channel images, so a
        hyperspectral cube [B, C, H, W] with C not in {1, 3} is mapped to
        a pseudo-RGB preview by sampling three representative bands spread
        across the spectrum (~R, G, B). Values are clamped to [0, 1].
        '''
        c = imgs.shape[1]
        if c not in (1, 3):
            idx = [round(f * (c - 1)) for f in (0.73, 0.5, 0.2)]  # ~R,G,B
            imgs = imgs[:, idx]
        return imgs.clamp(0.0, 1.0)

    def _preview(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, prefix):
        if (not trainer.is_global_zero or trainer.sanity_checking or batch_idx != 0
                or dataloader_idx != 0 or not self.every_n_epochs
                or (trainer.current_epoch != 0
                    and (trainer.current_epoch + 1) % self.every_n_epochs != 0)):
            return
        src, target = batch[:2]
        if isinstance(outputs, dict) and "preview_source" in outputs:
            src = outputs["preview_source"]
        h, w = src.shape[-2:]
        top, left = max(0, (h - self.max_size) // 2), max(0, (w - self.max_size) // 2)

        def patch(images):
            return images[:self.max_images, :, top:top + self.max_size, left:left + self.max_size]

        # USGS returns the prediction already computed by validation/test.
        # Other pipelines retain a bounded-patch fallback without another loader.
        prediction = outputs.get("image") if isinstance(outputs, dict) else None
        with torch.no_grad():
            reconst = patch(prediction) if prediction is not None else pl_module(patch(src))
            imgs = torch.stack([
                self._to_rgb(patch(src)), self._to_rgb(reconst),
                self._to_rgb(patch(target)),
            ], dim=1).flatten(0, 1).detach().cpu()
        directory = Path(trainer.log_dir or trainer.default_root_dir) / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(imgs, nrow=3),
            directory / f"{prefix}_{trainer.current_epoch}.png",
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
    ):
        self._preview(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "reconst")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
    ):
        self._preview(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "test")

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_dir = Path(trainer.log_dir or trainer.default_root_dir).joinpath('figures')
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor,
        batch: Any,
        batch_idx: int,
    ) -> None:

        images = outputs['image']
        filenames = outputs['filenames']
        for i in range(images.shape[0]):
            image = images[i].permute(1,2,0).detach().cpu().numpy()
            path = self.save_dir.joinpath(f'{filenames[i]}')
            write(path, image)
