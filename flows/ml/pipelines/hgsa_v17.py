import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from typing import Dict, Tuple
from flows.core import Logger
from flows.tools.utils import models
from ..metrics import PSNR, SSIM, DeltaE


# --- Loss Functions ---
class LogCoshLoss(nn.Module):

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        x = y_pred - y_true
        # Stable implementation of ln(cosh(x))
        loss = torch.abs(x) + F.softplus(-2.0 * torch.abs(x)) - math.log(2.0)
        return torch.mean(loss)


class GradLoss(nn.Module):

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # Spatial gradient penalty for edge preservation
        def gradient(x):
            r = F.pad(x, (0, 1, 0, 1))
            grad_x = r[:, :, :, 1:] - r[:, :, :, :-1]
            grad_y = r[:, :, 1:, :] - r[:, :, :-1, :]
            return grad_x[:, :, :, :-1], grad_y[:, :, :-1, :]

        grad_p_x, grad_p_y = gradient(pred)
        grad_t_x, grad_t_y = gradient(target)
        return F.l1_loss(grad_p_x, grad_t_x) + F.l1_loss(grad_p_y, grad_t_y)


# --- Pipeline HGSA v17 ---
class HSGAPipeline_v17(L.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 optimizer: str = 'adamw',
                 lr: float = 1e-3,
                 warmup_epochs: int = 10,
                 weight_decay: float = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Loss components
        self.mae_loss = nn.L1Loss()
        self.logcosh_loss = LogCoshLoss()
        self.grad_loss = GradLoss()

        # Metrics (Assuming these are imported correctly)
        self.psnr_metric = PSNR(data_range=1.0)
        self.ssim_metric = SSIM(data_range=1.0)
        self.de_metric = DeltaE()

        # Regularization weights
        self.w_tv = 0.05
        self.w_aux = 0.1
        self.w_color = 1.5  # Increased for dE optimization

        self.save_hyperparameters(ignore=['model'])

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d):
                    # Theorem 1: Low gain for hypernetwork stability
                    # Added 'orchestrator' and 'fusion_head' to the stable init list
                    hyper_nets = [
                        'xi_net', 'expert_heads', 'input_proj', 'orchestrator'
                    ]
                    if any(x in name for x in hyper_nets):
                        nn.init.xavier_uniform_(m.weight, gain=0.1)
                    else:
                        nn.init.kaiming_normal_(m.weight,
                                                mode="fan_out",
                                                nonlinearity="relu")

                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # MODEL_PATH = '.experiments/ggpd.hgsa_v16.huawei/logs/checkpoints/_last.ckpt'
        # models.load_model(self.model, 'model', MODEL_PATH)

        Logger.info(
            'HGSA_v17: Pipeline initialized with Orchestrator-aware weights.')

    def configure_optimizers(self):
        params_groups = {
            'encoder': [],
            'orchestra': [],
            'experts': [],
            'fusion': [],
        }

        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                params_groups['encoder'].append(param)
            elif any(x in name for x in ['xi_net', 'orchestrator']):
                params_groups['orchestra'].append(param)
            elif any(x in name for x in [
                    'expert_heads', 'chi_net', 'spectral_calibrator',
                    'mu_init', 'w_init'
            ]):
                params_groups['experts'].append(param)
            else:
                params_groups['fusion'].append(param)

        optimizer = optim.AdamW(
            [
                {
                    'params': params_groups['encoder'],
                    'lr': self.lr * 0.8
                },
                {
                    'params': params_groups['orchestra'],
                    'lr': self.lr * 1.0
                },
                {
                    'params': params_groups['experts'],
                    'lr': self.lr * 1.2,
                    'weight_decay': 1e-3
                },  # High LR for experts
                {
                    'params': params_groups['fusion'],
                    'lr': self.lr * 1.0
                },
            ],
            weight_decay=self.weight_decay)

        # Scheduler Logic (Preserved from your v16 request)
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        self.scheduler_switch_epoch = int(self.trainer.max_epochs * 0.8)

        total_steps_s1 = self.scheduler_switch_epoch * steps_per_epoch
        t_0_steps = max(1, total_steps_s1 // 3)

        # self.scheduler_1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        # optimizer, T_0=t_0_steps, T_mult=1, eta_min=self.lr * 0.01)

        self.scheduler_1 = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.scheduler_switch_epoch,
            total_steps=self.scheduler_switch_epoch * steps_per_epoch,
            pct_start=0.15,
            div_factor=10,
            final_div_factor=100)
        self.scheduler_2 = optim.lr_scheduler.ExponentialLR(optimizer,
                                                            gamma=0.97)

        return {"optimizer": optimizer}

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        if self.current_epoch < self.scheduler_switch_epoch:
            self.scheduler_1.step()
        else:
            if self.trainer.is_last_batch:
                self.scheduler_2.step()

    def total_variation_loss(self, img: torch.Tensor) -> torch.Tensor:
        diff_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        diff_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        src, tgt = batch

        # HGSA_v17 returns (main_res, aux_res) during training
        # We use self.model(src) directly instead of self(src) to avoid the clamp in forward()
        main_out, aux_out = self(src)

        loss_mae = self.mae_loss(main_out, tgt)
        loss_color = self.logcosh_loss(main_out, tgt)
        loss_grad = self.grad_loss(main_out, tgt)

        main_out_c = torch.clamp(main_out, 0.0, 1.0)
        # Using a slightly higher weight for SSIM to force the Fusion Head to learn details
        ssim_val = self.ssim_metric(main_out_c, tgt)
        loss_ssim = 1.0 - torch.clamp(ssim_val, 0., 1.)

        loss_aux = self.mae_loss(aux_out, tgt)
        loss_tv = self.total_variation_loss(main_out)

        w_grad = 1.0 if self.current_epoch < self.warmup_epochs else 0.7
        a = 0.1 if self.current_epoch < self.warmup_epochs else 0.9

        total_loss = a * loss_mae + 0.15 * loss_ssim + (1 - a) * loss_aux
        #total_loss = (
           # loss_mae + self.w_color * loss_color +  # Targeted dE optimization
           # w_grad * loss_grad + 0.2 * loss_ssim + self.w_aux * loss_aux +
           # self.w_tv * loss_tv)

        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_color', loss_color, prog_bar=False)
        self.log('train_grad', loss_grad, prog_bar=False)
        self.log('train_tv', loss_tv, prog_bar=False)

        with torch.no_grad():
            psnr_val = self.psnr_metric(main_out_c, tgt)
            de_val = self.de_metric(main_out_c, tgt)
            self.log('train_psnr', psnr_val, prog_bar=True)
            self.log('train_ssim', ssim_val, prog_bar=True)
            self.log('train_de', de_val, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        src, tgt = batch
        y = self(src)  # In eval, model returns only main_res
        y = torch.clamp(y, 0.0, 1.0)

        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('val_psnr', psnr_val, prog_bar=True)
        self.log('val_ssim', ssim_val, prog_bar=True)
        self.log('val_de', de_val, prog_bar=True)
        self.log('val_loss', de_val, prog_bar=True)

        return psnr_val

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        src, tgt = batch
        y = self(src)  # In eval, model returns only main_res
        y = torch.clamp(y, 0.0, 1.0)

        psnr_val = self.psnr_metric(y, tgt)
        ssim_val = self.ssim_metric(y, tgt)
        de_val = self.de_metric(y, tgt).mean()

        self.log('test_psnr', psnr_val, prog_bar=True)
        self.log('test_ssim', ssim_val, prog_bar=True)
        self.log('test_de', de_val, prog_bar=True)
        self.log('test_loss', de_val, prog_bar=True)

        return psnr_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard inference
        return self.model(src=x)['res']
