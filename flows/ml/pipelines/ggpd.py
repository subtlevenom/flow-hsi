import os
import random
import statistics
from typing import List
from einops import rearrange
import torch
from torch import nn
import lightning as L
from torch import optim
import torch.nn.functional as F
import torchvision
import time
from flows.tools.utils import text
from flows.tools.utils import models
from flows.core import Logger
from flows.ml.losses import GPDFLoss
from ..models import Flow
from ..metrics import (PSNR, SSIM, SAM, DeltaE)
from flows.ml.layers.sep_gpd import MultivariateNormal


class GGPDPipeline(L.LightningModule):

    def __init__(self,
                 model: Flow,
                 optimizer: str = 'adam',
                 lr: float = 1e-3,
                 weight_decay: float = 0,
                 warmup_epochs: int = 0,
                 hgsa_ckpt: str = None,
                 sam_weight: float = 0.1,
                 metrics_channels: List[int] = [0, 1, 2]) -> None:
        super(GGPDPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        # Warm-up: freeze the pretrained HGSA color-transport core for the
        # first ``warmup_epochs`` epochs so the surrounding MSAB encoder/
        # decoder can adapt to it before the whole network is fine-tuned.
        self.warmup_epochs = warmup_epochs
        self.hgsa_ckpt = hgsa_ckpt
        # Spectral-angle (SAM) loss weight — key spectral-fidelity term for
        # the HSI task; MAE alone matches per-band intensity but ignores
        # the shape of the spectral signature.
        self.sam_weight = sam_weight
        self._hgsa_frozen = False
        self.ggpd_loss = GPDFLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.de_metric = DeltaE()
        self.sam_metric = SAM()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.metrics_channels = metrics_channels

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str) -> None:
        '''
        Initialize model weights, load the pretrained HGSA color-transport
        core, and (optionally) freeze it for the warm-up epochs so the
        surrounding MSAB encoder/decoder can adapt to the frozen core
        before the whole network is fine-tuned jointly.
        '''
        if stage == 'fit' or stage is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d):
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

            # Load the pretrained HGSA core over the freshly-initialized
            # weights, then freeze it for the warm-up phase. Only done for
            # training — for test/predict the full checkpoint is restored
            # by Lightning and must not be partially overwritten here.
            # Skipped for models without a discrete `hgsa` layer (e.g. the
            # USGS pyramid, whose per-level cores are trained jointly).
            has_hgsa = (hasattr(self.model, 'layers')
                        and 'hgsa' in self.model.layers)
            if has_hgsa and self.hgsa_ckpt and os.path.isfile(self.hgsa_ckpt):
                models.load_model(
                    self.model.layers.hgsa, '_model', self.hgsa_ckpt)
                Logger.info(
                    f'Loaded pretrained HGSA core from {self.hgsa_ckpt}.')
            elif has_hgsa:
                Logger.info(
                    'No HGSA checkpoint found; training HGSA from scratch.')

            if has_hgsa and self.warmup_epochs > 0:
                self._set_hgsa_frozen(True)
                Logger.info(
                    f'HGSA core frozen for the first {self.warmup_epochs} '
                    f'warm-up epoch(s).')

            Logger.info('Initialized model weights with isp pipeline.')

    def _set_hgsa_frozen(self, frozen: bool) -> None:
        '''Freeze/unfreeze the pretrained HGSA color-transport core.'''
        if not (hasattr(self.model, 'layers')
                and 'hgsa' in self.model.layers):
            return
        for p in self.model.layers.hgsa.parameters():
            p.requires_grad = not frozen
        self._hgsa_frozen = frozen

    def on_train_epoch_start(self) -> None:
        # Unfreeze the HGSA core once the warm-up phase is over so the
        # whole network is fine-tuned jointly.
        if self._hgsa_frozen and self.current_epoch >= self.warmup_epochs:
            self._set_hgsa_frozen(False)
            Logger.info(f'Unfroze HGSA core at epoch {self.current_epoch}.')

    def spectral_angle_loss(self, pred: torch.Tensor,
                            tgt: torch.Tensor) -> torch.Tensor:
        '''Mean spectral angle (radians) across the spectral dimension.

        Central spectral-fidelity objective for the HSI task. Fully
        differentiable and guarded against the 0/0 (zero-spectrum) and
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
        '''Mean Relative Absolute Error — the canonical NTIRE spectral
        reconstruction metric.

        Normalizes the per-pixel error by target magnitude, so dark
        (low-reflectance) bands are weighted comparably to bright ones
        — unlike plain MAE. ``eps`` floors the denominator to keep the
        gradient finite where the target is near zero (paired with the
        Trainer's gradient_clip_val for stability at batch_size=1).
        '''
        return (torch.abs(pred - tgt) / (tgt.abs() + eps)).mean()

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
        # eta_min. (Previously T_0=500 with max_epochs<500 meant the cosine
        # never completed a cycle and the LR never annealed.)
        t_max = getattr(self.trainer, 'max_epochs', None) or 500
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_mrae"
        }

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pred = self.model(src=x, tgt=y)
        return pred['res']

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        out = self(src, tgt)

        # Deep supervision: the USGS pyramid returns per-scale outputs
        # (fine, mid, coarse) during training. Supervise each scale with
        # MRAE against the correspondingly down-sampled target.
        if isinstance(out, (tuple, list)):
            y = out[0].to(torch.float32)
            mrae_loss = self.mrae_loss(y, tgt)
            ds_weights = (0.5, 0.25)
            for yi, wi in zip(out[1:], ds_weights):
                yi = yi.to(torch.float32)
                tgt_i = F.interpolate(tgt, size=yi.shape[-2:],
                                      mode='bilinear', align_corners=False)
                mrae_loss = mrae_loss + wi * self.mrae_loss(yi, tgt_i)
        else:
            y = out.to(torch.float32)
            mrae_loss = self.mrae_loss(y, tgt)

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        sam_ang = self.spectral_angle_loss(y, tgt)
        # MRAE is the primary NTIRE objective; SAM enforces spectral shape.
        loss = mrae_loss + self.sam_weight * sam_ang

        self.log('mae', mae_loss, prog_bar=True, logger=True)
        self.log('mrae', mrae_loss, prog_bar=True, logger=True)
        self.log('psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('sam', sam_loss, prog_bar=True, logger=True)
        self.log('hgsa_frozen', float(self._hgsa_frozen),
                 prog_bar=True, logger=True)
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
        loss = mrae_loss + self.sam_weight * sam_loss

        self.log('val_mae', mae_loss, prog_bar=True, logger=True)
        self.log('val_mrae', mrae_loss, prog_bar=True, logger=True)
        self.log('val_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('val_sam', sam_loss, prog_bar=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    # Bare-Parameter gate scalars whose magnitude tells us whether a gated
    # (near-identity at init) capacity module is actually coming online. Keyed
    # by a short log name -> predicate on the fully-qualified parameter name.
    # A model without these gates simply logs nothing (all no-op).
    _GATE_SPECS = {
        'gate/out': lambda n: n.endswith('.out_gate'),
        'gate/kst': lambda n: n.endswith('.kst_gamma'),
        'gate/head_spatial': lambda n: n.endswith('.head_gamma'),
        'gate/offset_scale': lambda n: n.endswith('.offset_scale'),
        'gate/spectral_smooth': lambda n: n.endswith('.smooth_gamma'),
        'gate/kan_in_scale': lambda n: n.endswith('.in_scale'),
        'gate/post_crossband': lambda n: n.endswith('.g_cb'),
        'gate/post_spatial': lambda n: n.endswith('.g_sp'),
        'gate/fine_refine': lambda n: 'refine' in n and n.endswith('.gamma'),
        'gate/lccm': lambda n: 'lccm' in n and n.endswith('.gamma'),
    }

    def on_validation_epoch_end(self) -> None:
        '''Log the mean magnitude of each gated-capacity scalar.

        For gated residuals ``y + gamma * f(x)`` the module's contribution — and
        its learning rate — scale with ``|gamma|``. Tracking these over epochs
        shows whether the A-E capacity levers are activating (gates growing) or
        staying inert (stuck at init), which for a gated hypernetwork is far more
        diagnostic than the early-epoch val_psnr level.
        '''
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
        loss = mrae_loss + self.sam_weight * sam_loss

        self.log('test_mae', mae_loss, prog_bar=True, logger=True)
        self.log('test_mrae', mrae_loss, prog_bar=True, logger=True)
        self.log('test_psnr', psnr_loss, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_loss, prog_bar=True, logger=True)
        self.log('test_sam', sam_loss, prog_bar=True, logger=True)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    sum_mae = 0
    sum_psnr = 0
    sum_ssim = 0
    sum_sam = 0
    sum_mrae = 0
    start_time = 0

    def predict_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.start_time = time.perf_counter()

        src, tgt, name = batch
        y = self(src, tgt)
        elapsed = time.perf_counter() - self.start_time

        mae_loss = self.mae_loss(y, tgt)
        mrae_loss = self.mrae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)

        self.sum_mae += mae_loss
        self.sum_psnr += psnr_loss
        self.sum_ssim += ssim_loss
        self.sum_sam += sam_loss
        self.sum_mrae += mrae_loss
        n = 1 + batch_idx

        text.print_json({
            name[0]: {
                'CUR': {
                    'mae': mae_loss.item(),
                    'mrae': mrae_loss.item(),
                    'psnr': psnr_loss.item(),
                    'ssim': ssim_loss.item(),
                    'sam': sam_loss.item(),
                },
                'AVG': {
                    'mae': self.sum_mae.item() / n,
                    'mrae': self.sum_mrae.item() / n,
                    'psnr': self.sum_psnr.item() / n,
                    'ssim': self.sum_ssim.item() / n,
                    'sam': self.sum_sam.item() / n,
                },
                'TIME': elapsed / n,
            },
        })

        return {'loss': mrae_loss}
