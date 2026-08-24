from typing import List

import torch
from torch import nn
import lightning as L
from torch import optim
import torch.nn.functional as F
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim_fn)

from flows.core import Logger
from ..models import Flow
from ..metrics import (PSNR, SSIM, SAM, DeltaE)


class USGSExpertPipeline(L.LightningModule):
    '''Dual-branch training pipeline for the USGS expert restorer.

    The :class:`USGSExpertPyramid` model exposes two explicit branch
    outputs during training:

    * ``struct`` — the sharpness / structural-enhancement branch. Trained
      with an SSIM + gradient (high-frequency) objective so it learns edges,
      texture and local contrast — the quantities SSIM rewards.
    * ``color`` — the color-matching branch. Trained with an L1 + spectral
      angle objective so it learns per-band intensity and the shape of the
      spectral signature — the quantities PSNR / dE / SAM reward.

    The fused ``main`` output (plus its coarse pyramid ``aux`` scales) is
    supervised with the canonical NTIRE MRAE + SAM objective so the two
    specialised branches are reconciled into the final reconstruction.
    '''

    def __init__(self,
                 model: Flow,
                 optimizer: str = 'adam',
                 lr: float = 1e-3,
                 weight_decay: float = 0,
                 sam_weight: float = 0.1,
                 struct_weight: float = 0.5,
                 color_weight: float = 0.5,
                 kl_weight: float = 1e-4,
                 ds_weights: List[float] = [0.5, 0.25],
                 metrics_channels: List[int] = [0, 1, 2]) -> None:
        super(USGSExpertPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        # Central spectral-fidelity weight for the fused/main reconstruction.
        self.sam_weight = sam_weight
        # Branch-loss weights: how strongly each specialised branch is pulled
        # toward its own objective, on top of the fused reconstruction.
        self.struct_weight = struct_weight
        self.color_weight = color_weight
        # KL weight for the expert-latent Gaussians (near-zero — the latent
        # is a light regulariser, not a generative bottleneck).
        self.kl_weight = kl_weight
        # Deep-supervision weights for the coarse pyramid ``aux`` scales.
        self.ds_weights = ds_weights

        self.mae_loss = nn.L1Loss(reduction='mean')
        self.de_metric = DeltaE()
        self.sam_metric = SAM()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.metrics_channels = metrics_channels

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        '''Initialize model weights with the standard ISP scheme.

        Generic Conv/Linear/BatchNorm re-init only; the model's near-identity
        gates (``struct_gate``, ``color_gate``, ``out_gate`` and the various
        ``gamma`` scalars) are bare Parameters and survive this pass, so the
        network starts close to identity and grows capacity as the gates open.
        '''
        if stage == 'fit' or stage is None:
            for m in self.model.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            Logger.info('Initialized model weights with isp pipeline.')

    def spectral_angle_loss(self, pred: torch.Tensor,
                            tgt: torch.Tensor) -> torch.Tensor:
        '''Mean spectral angle (radians) across the spectral dimension.

        Fully differentiable and guarded against the 0/0 (zero-spectrum) and
        arccos(±1) singularities so gradients stay finite.
        '''
        p = pred.flatten(2)                       # [B, C, H*W]
        t = tgt.flatten(2)
        dot = (p * t).sum(dim=1)                   # [B, H*W]
        denom = p.norm(dim=1) * t.norm(dim=1) + 1e-8
        cos = (dot / denom).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.arccos(cos).mean()

    def mrae_loss(self, pred: torch.Tensor, tgt: torch.Tensor,
                  eps: float = 1e-3) -> torch.Tensor:
        '''Mean Relative Absolute Error — the canonical NTIRE metric.'''
        return (torch.abs(pred - tgt) / (tgt.abs() + eps)).mean()

    def gradient_l1(self, pred: torch.Tensor,
                    tgt: torch.Tensor) -> torch.Tensor:
        '''First-order gradient (high-frequency) L1 — the sharpness term.

        Penalises differences in horizontal/vertical finite differences so
        the structural branch is pushed to reproduce edges and texture that
        a flat intensity loss ignores.
        '''
        dx_p = pred[..., :, 1:] - pred[..., :, :-1]
        dx_t = tgt[..., :, 1:] - tgt[..., :, :-1]
        dy_p = pred[..., 1:, :] - pred[..., :-1, :]
        dy_t = tgt[..., 1:, :] - tgt[..., :-1, :]
        return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()

    def structural_loss(self, pred: torch.Tensor,
                        tgt: torch.Tensor) -> torch.Tensor:
        '''SSIM (structure/sharpness) + gradient objective for the structural
        branch. Uses the differentiable functional SSIM to avoid the stateful
        metric accumulating across steps.'''
        ssim = ssim_fn(pred, tgt, data_range=1.0)
        return (1.0 - ssim) + self.gradient_l1(pred, tgt)

    def color_loss(self, pred: torch.Tensor,
                   tgt: torch.Tensor) -> torch.Tensor:
        '''L1 (intensity) + spectral-angle objective for the color branch —
        the quantities PSNR / dE / SAM reward.'''
        return self.mae_loss(pred, tgt) + self.spectral_angle_loss(pred, tgt)

    def kl_gaussian(self, mu: torch.Tensor,
                    logvar: torch.Tensor) -> torch.Tensor:
        '''KL(N(mu, exp(logvar)) || N(0, 1)) — light latent regulariser.'''
        return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1.0 - logvar)

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.weight_decay)
        else:
            raise ValueError(
                f'unsupported optimizer_type: {self.optimizer_type}')
        # Single cosine decay over the whole run so the LR actually reaches
        # eta_min.
        t_max = getattr(self.trainer, 'max_epochs', None) or 500
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_mrae"
        }

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        # In training the model returns the dual-branch dict; in eval it
        # returns the fused tensor. Flow binds the whole return to ``res``.
        pred = self.model(src=x, tgt=y)
        return pred['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        out = self(src, tgt)

        main = out['main'].to(torch.float32)
        struct = out['struct'].to(torch.float32)
        color = out['color'].to(torch.float32)

        # --- Fused reconstruction (MRAE + SAM) with deep supervision on the
        #     coarse pyramid scales. ---
        mrae = self.mrae_loss(main, tgt)
        for yi, wi in zip(out.get('aux', []), self.ds_weights):
            yi = yi.to(torch.float32)
            tgt_i = F.interpolate(tgt, size=yi.shape[-2:],
                                  mode='bilinear', align_corners=False)
            mrae = mrae + wi * self.mrae_loss(yi, tgt_i)
        recon_loss = mrae + self.sam_weight * self.spectral_angle_loss(main, tgt)

        # --- Structural / sharpness branch (SSIM + gradient). ---
        struct_loss = self.structural_loss(struct, tgt)

        # --- Color-matching branch (L1 + spectral angle). ---
        color_loss = self.color_loss(color, tgt)

        # --- Expert-latent KL regulariser. ---
        kl = self.kl_gaussian(out['mu'].to(torch.float32),
                              out['logvar'].to(torch.float32))

        loss = (recon_loss
                + self.struct_weight * struct_loss
                + self.color_weight * color_loss
                + self.kl_weight * kl)

        # Monitoring metrics on the fused output.
        mae_loss = self.mae_loss(main, tgt)
        psnr_loss = self.psnr_metric(main, tgt)
        ssim_loss = self.ssim_metric(main, tgt)
        sam_loss = self.sam_metric(main, tgt)

        self.log('mae', mae_loss, prog_bar=True, logger=True)
        self.log('mrae', mrae, prog_bar=True, logger=True)
        self.log('psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('sam', sam_loss, prog_bar=True, logger=True)
        self.log('struct_loss', struct_loss, prog_bar=False, logger=True)
        self.log('color_loss', color_loss, prog_bar=False, logger=True)
        self.log('kl', kl, prog_bar=False, logger=True)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src, tgt).to(torch.float32)

        mae_loss = self.mae_loss(y, tgt)
        mrae_loss = self.mrae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        # DeltaE needs 3-channel RGB (rgb_to_lab) — evaluate on the slice.
        ch = self.metrics_channels
        de_loss = self.de_metric(y[:, ch], tgt[:, ch])
        loss = mrae_loss + self.sam_weight * sam_loss

        self.log('val_mae', mae_loss, prog_bar=True, logger=True)
        self.log('val_mrae', mrae_loss, prog_bar=True, logger=True)
        self.log('val_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('val_sam', sam_loss, prog_bar=True, logger=True)
        self.log('val_de', de_loss, prog_bar=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    # Bare-Parameter gate scalars whose magnitude tells us whether a gated
    # (near-identity at init) capacity module is actually coming online. Keyed
    # by a short log name -> predicate on the fully-qualified parameter name.
    _GATE_SPECS = {
        'gate/out': lambda n: n.endswith('.out_gate'),
        'gate/struct': lambda n: n.endswith('.struct_gate'),
        'gate/color': lambda n: n.endswith('.color_gate'),
        'gate/mix': lambda n: 'mix' in n and n.endswith('.gamma'),
        'gate/fusion': lambda n: 'fusion' in n and n.endswith('.gamma'),
    }

    def on_validation_epoch_end(self) -> None:
        '''Log the mean magnitude of each gated-capacity scalar so we can see
        whether the branch gates are activating (growing) or staying inert.'''
        sums = {k: [0.0, 0] for k in self._GATE_SPECS}
        for name, p in self.model.named_parameters():
            for key, match in self._GATE_SPECS.items():
                if match(name):
                    sums[key][0] += p.detach().abs().mean().item()
                    sums[key][1] += 1
        for key, (total, count) in sums.items():
            if count:
                self.log(key, total / count, prog_bar=False, logger=True)

    def test_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src, tgt).to(torch.float32)

        mae_loss = self.mae_loss(y, tgt)
        mrae_loss = self.mrae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        ch = self.metrics_channels
        de_loss = self.de_metric(y[:, ch], tgt[:, ch])
        loss = mrae_loss + self.sam_weight * sam_loss

        self.log('test_mae', mae_loss, prog_bar=True, logger=True)
        self.log('test_mrae', mrae_loss, prog_bar=True, logger=True)
        self.log('test_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('test_sam', sam_loss, prog_bar=True, logger=True)
        self.log('test_de', de_loss, prog_bar=True, logger=True)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}
