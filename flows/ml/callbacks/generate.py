from typing import Any
from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
import torch.nn.functional as F
import torch
import torchvision
import os
from flows.tools.files import write


class GenerateCallback(Callback):
    def __init__(
            self,
            every_n_epochs=1
        ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.input_imgs = None
        self.save_dir = None
        self.target_imgs = None

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

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.val_dataloaders
        self.input_imgs, self.target_imgs = next(iter(dataloader))
        self.input_imgs = self.input_imgs.to(pl_module.device)
        self.target_imgs = self.target_imgs.to(pl_module.device)
        self.save_dir = os.path.join(trainer.log_dir, 'figures')

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(self.input_imgs)
                pl_module.train()

            # Map to pseudo-RGB so hyperspectral (e.g. 31-band) cubes are
            # saveable — PIL/torchvision only handle 1- or 3-channel images.
            input_imgs = self._to_rgb(self.input_imgs)
            target_imgs = self._to_rgb(self.target_imgs)
            reconst_imgs = self._to_rgb(reconst_imgs)
            imgs = torch.stack([input_imgs, reconst_imgs, target_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3)
            # Save image
            save_path = os.path.join(self.save_dir, f"reconst_{trainer.current_epoch}.png")
            os.makedirs(self.save_dir, exist_ok=True)
            torchvision.utils.save_image(grid, save_path)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.test_dataloaders
        self.input_imgs, self.target_imgs = next(iter(dataloader))
        self.input_imgs = self.input_imgs.to(pl_module.device)
        self.target_imgs = self.target_imgs.to(pl_module.device)
        self.save_dir = os.path.join(trainer.log_dir, 'figures')

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(self.input_imgs)
                pl_module.train()
            # Plot and add to tensorboard

            input_imgs = self._to_rgb(self.input_imgs)
            target_imgs = self._to_rgb(self.target_imgs)
            reconst_imgs = self._to_rgb(reconst_imgs)
            imgs = torch.stack([input_imgs, reconst_imgs, target_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3)
            # Save image
            save_path = os.path.join(self.save_dir, f"test_{trainer.current_epoch}.png")
            os.makedirs(self.save_dir, exist_ok=True)
            torchvision.utils.save_image(grid, save_path)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_dir = torch.Path(trainer.log_dir).joinpath('figures')
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
