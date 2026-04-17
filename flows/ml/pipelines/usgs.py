import time
import math
from typing import List, Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim

from flows.core import Logger
from ..metrics import DeltaE, PSNR, SAM, SSIM
from ..models import Flow
from lightning.pytorch.callbacks import StochasticWeightAveraging


class USGSPipeline(L.LightningModule):
    """Lightning module for reference-free oscillation model training.

    The module is API-compatible with multiple model signatures:
    - new: model(src, return_aux=...)
    - legacy: model(src=..., tgt=...) -> {"res": tensor}
    - fallback: model.layers.usgs(src)
    """

    def __init__(
        self,
        model: Flow,
        optimizer: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 5e-5,
        warmup_epochs: int = 10,
        metrics_channels: Optional[List[int]] = None,
        lambda_ssim: float = 0.15,
        lambda_lab: float = 0.2,
        lambda_de: float = 0.15,
        lambda_lab_moment: float = 0.04,
        lambda_sat_guard: float = 0.02,
        lambda_param_reg: float = 0.01,
        stage1_epochs: int = 30,
        stage2_epochs: int = 180,
        stage3_epochs: int = 320,
        min_lr_stage2: float = 5e-5,
        min_lr_stage3: float = 8e-6,
    ) -> None:
        super().__init__()

        self.model = model
        self.optimizer_type = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.lambda_ssim = lambda_ssim
        self.lambda_lab = lambda_lab
        self.lambda_de = lambda_de
        self.lambda_lab_moment = lambda_lab_moment
        self.lambda_sat_guard = lambda_sat_guard
        self.lambda_param_reg = lambda_param_reg
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.min_lr_stage2 = min_lr_stage2
        self.min_lr_stage3 = min_lr_stage3

        self.base_lambda_ssim = lambda_ssim
        self.base_lambda_lab = lambda_lab
        self.base_lambda_de = lambda_de
        self.base_lambda_lab_moment = lambda_lab_moment
        self.base_lambda_sat_guard = lambda_sat_guard
        self.base_lambda_param_reg = lambda_param_reg

        self.mae_loss = nn.L1Loss(reduction="mean")
        self.de_metric = DeltaE()
        self.sam_metric = SAM()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))

        if metrics_channels is None:
            metrics_channels = [0, 1, 2]
        self.metrics_channels = metrics_channels

        self.save_hyperparameters(ignore=["model"])

        self.sum_mae = 0.0
        self.sum_psnr = 0.0
        self.sum_ssim = 0.0
        self.sum_sam = 0.0
        self.sum_de = 0.0
        self.start_time = 0.0

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        Logger.info("Initialized model weights with Kaiming init.")

    @staticmethod
    def get_swa_callback(swa_epoch_start: int = 200, swa_lrs: float = 1e-5) -> StochasticWeightAveraging:
        """Get SWA callback for improved convergence. Use in trainer initialization:
        
        Example:
            trainer = Trainer(
                callbacks=[
                    USGSPipeline.get_swa_callback(swa_epoch_start=200),
                    ModelCheckpoint(monitor='val_psnr', mode='max'),
                ],
                ...
            )
        """
        return StochasticWeightAveraging(swa_lrs=swa_lrs, swa_epoch_start=swa_epoch_start)

    def configure_optimizers(self):
        use_foreach = torch.cuda.is_available()
        if self.optimizer_type == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.99),
                foreach=use_foreach,
            )
        elif self.optimizer_type == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.99),
                foreach=use_foreach,
            )
        elif self.optimizer_type == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"unsupported optimizer_type: {self.optimizer_type}")

        return optimizer

    @staticmethod
    def _cosine_interp(start: float, end: float, progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        return end + 0.5 * (start - end) * (1.0 + math.cos(progress * math.pi))

    def _lambda_warmup_scale(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        progress = (epoch + 1) / max(self.warmup_epochs, 1)
        return float(min(max(progress, 0.0), 1.0))

    def _get_schedule_values(self, epoch: int):
        if epoch < self.stage1_epochs:
            return {
                "stage": 1,
                "lr": self.lr,
                "weight_decay": 1e-4,
                "lambda_ssim": self.base_lambda_ssim,
                "lambda_lab": max(self.base_lambda_lab, 0.25),
                "lambda_de": max(self.base_lambda_de, 0.10),
                "lambda_lab_moment": max(self.base_lambda_lab_moment, 0.05),
                "lambda_sat_guard": max(self.base_lambda_sat_guard, 0.02),
            }

        if epoch < self.stage2_epochs:
            progress = (epoch - self.stage1_epochs) / max(self.stage2_epochs - self.stage1_epochs, 1)
            return {
                "stage": 2,
                "lr": self._cosine_interp(self.lr, self.min_lr_stage2, progress),
                "weight_decay": 5e-5,
                "lambda_ssim": self.base_lambda_ssim,
                "lambda_lab": self.base_lambda_lab,
                "lambda_de": self.base_lambda_de,
                "lambda_lab_moment": self.base_lambda_lab_moment,
                "lambda_sat_guard": self.base_lambda_sat_guard,
            }

        progress = (epoch - self.stage2_epochs) / max(self.stage3_epochs - self.stage2_epochs, 1)
        return {
            "stage": 3,
            "lr": self._cosine_interp(self.min_lr_stage2, self.min_lr_stage3, progress),
            "weight_decay": 1e-5,
            "lambda_ssim": self.base_lambda_ssim,
            "lambda_lab": self.base_lambda_lab + 0.05,
            "lambda_de": self.base_lambda_de + 0.02,
            "lambda_lab_moment": self.base_lambda_lab_moment + 0.01,
            "lambda_sat_guard": self.base_lambda_sat_guard,
        }

    def on_train_epoch_start(self) -> None:
        schedule = self._get_schedule_values(self.current_epoch)
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]

        for group in optimizer.param_groups:
            group["lr"] = schedule["lr"]
            group["weight_decay"] = schedule["weight_decay"]

        self.lambda_ssim = schedule["lambda_ssim"]
        self.lambda_lab = schedule["lambda_lab"]
        warmup_scale = self._lambda_warmup_scale(self.current_epoch)
        self.lambda_de = schedule["lambda_de"] * warmup_scale
        self.lambda_lab_moment = schedule["lambda_lab_moment"] * warmup_scale
        self.lambda_sat_guard = schedule["lambda_sat_guard"] * warmup_scale

        self.log("sched_stage", float(schedule["stage"]), prog_bar=True, logger=True)
        self.log("sched_lr", float(schedule["lr"]), prog_bar=True, logger=True)
        self.log("sched_wd", float(schedule["weight_decay"]), prog_bar=False, logger=True)
        self.log("sched_lambda_lab", float(self.lambda_lab), prog_bar=False, logger=True)
        self.log("sched_lambda_de", float(self.lambda_de), prog_bar=False, logger=True)
        self.log("sched_lambda_lab_moment", float(self.lambda_lab_moment), prog_bar=False, logger=True)
        self.log("sched_lambda_sat", float(self.lambda_sat_guard), prog_bar=False, logger=True)
        self.log("sched_lambda_param_reg", float(self.lambda_param_reg), prog_bar=False, logger=True)
        self.log("sched_lambda_warmup", float(warmup_scale), prog_bar=False, logger=True)

    @staticmethod
    def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
        a = 0.055
        return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)).pow(2.4))

    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb_lin = self._srgb_to_linear(rgb.clamp(0.0, 1.0))
        r = rgb_lin[:, 0:1]
        g = rgb_lin[:, 1:2]
        b = rgb_lin[:, 2:3]

        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        xn, yn, zn = 0.95047, 1.0, 1.08883
        x = x / xn
        y = y / yn
        z = z / zn

        eps = 216.0 / 24389.0
        k = 24389.0 / 27.0

        def f(t: torch.Tensor) -> torch.Tensor:
            return torch.where(t > eps, t.pow(1.0 / 3.0), (k * t + 16.0) / 116.0)

        fx, fy, fz = f(x), f(y), f(z)
        l = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return torch.cat([l, a, b], dim=1)

    def _lab_l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self._rgb_to_lab(pred) - self._rgb_to_lab(target)).abs().mean()

    def _delta_e76_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = self._rgb_to_lab(pred) - self._rgb_to_lab(target)
        return torch.sqrt((delta * delta).sum(dim=1) + 1e-12).mean()

    def _lab_moment_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = self._rgb_to_lab(pred)
        tgt_lab = self._rgb_to_lab(target)

        dims = (2, 3)
        mu_pred = pred_lab.mean(dim=dims)
        mu_tgt = tgt_lab.mean(dim=dims)
        std_pred = pred_lab.std(dim=dims, unbiased=False)
        std_tgt = tgt_lab.std(dim=dims, unbiased=False)

        return F.l1_loss(mu_pred, mu_tgt) + 0.5 * F.l1_loss(std_pred, std_tgt)

    def _saturation_guard_loss(self, pred: torch.Tensor) -> torch.Tensor:
        return (F.relu(pred - 0.98) + F.relu(0.02 - pred)).mean()

    def _parse_batch(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                src, tgt = batch
                return src, tgt, None, None
            if len(batch) == 3:
                src, tgt, third = batch
                if torch.is_tensor(third):
                    return src, tgt, third, None
                return src, tgt, None, third
            if len(batch) >= 4:
                src, tgt, ref, names = batch[0], batch[1], batch[2], batch[3]
                return src, tgt, ref, names
        raise ValueError("Unsupported batch format.")

    def _model_forward(
        self,
        src: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        # New oscillation API (reference-free)
        try:
            return self.model(
                src,
                return_aux=return_aux,
            )
        except TypeError:
            pass

        # Legacy API
        try:
            out = self.model(src=src, tgt=ref)
            if isinstance(out, dict) and "res" in out:
                return (out["res"], None) if return_aux else out["res"]
            return (out, None) if return_aux else out
        except TypeError:
            pass

        # Fallback
        out = self.model.layers.usgs(src)
        return (out, None) if return_aux else out

    def _parameter_reg_loss(self, aux) -> torch.Tensor:
        """Regularize learned Gaussian parameters: amplitude, center, sigma.
        
        Encourages:
        - Sparse amplitude: prefer selective basis usage
        - Diverse centers: prevent duplicate basis functions
        - Stable sigma: avoid extreme values
        """
        if aux is None:
            return torch.zeros((), device=self.device)
        amplitude = aux.get("amplitude")
        center = aux.get("center")
        if amplitude is None or center is None:
            return torch.zeros((), device=self.device)
        
        # Sparsity: encourage small amplitudes
        reg_amplitude = amplitude.abs().mean()
        
        # Diversity: penalize low variance in centers (want them spread out)
        # Average over batch and spatial dims, keep M dimension
        center_var = center.var(dim=(0, 2, 3)).mean()  # Compute variance per basis
        reg_center_diversity = 1.0 / (center_var + 1e-6)  # Penalize when centered cluster
        
        return reg_amplitude + 0.1 * reg_center_diversity

    def _composite_loss(self, pred: torch.Tensor, tgt: torch.Tensor, aux=None):
        mae_loss = self.mae_loss(pred, tgt)
        ssim_score = self.ssim_metric(pred, tgt)
        pred_lab = self._rgb_to_lab(pred)
        tgt_lab = self._rgb_to_lab(tgt)
        lab_delta = pred_lab - tgt_lab

        lab_loss = lab_delta.abs().mean()
        param_reg_loss = self._parameter_reg_loss(aux)
        de_loss = torch.sqrt((lab_delta * lab_delta).sum(dim=1) + 1e-12).mean()

        dims = (2, 3)
        mu_pred = pred_lab.mean(dim=dims)
        mu_tgt = tgt_lab.mean(dim=dims)
        std_pred = pred_lab.std(dim=dims, unbiased=False)
        std_tgt = tgt_lab.std(dim=dims, unbiased=False)
        lab_moment_loss = F.l1_loss(mu_pred, mu_tgt) + 0.5 * F.l1_loss(std_pred, std_tgt)

        sat_guard_loss = self._saturation_guard_loss(pred)

        loss = (
            mae_loss
            + self.lambda_ssim * (1 - ssim_score)
            + self.lambda_lab * lab_loss
            + self.lambda_de * de_loss
            + self.lambda_lab_moment * lab_moment_loss
            + self.lambda_sat_guard * sat_guard_loss
            + self.lambda_param_reg * param_reg_loss
        )
        return (
            loss,
            mae_loss,
            ssim_score,
            lab_loss,
            param_reg_loss,
            de_loss,
            lab_moment_loss,
            sat_guard_loss,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self._model_forward(
            src=x,
            ref=y,
            return_aux=False,
        )

    def training_step(self, batch, batch_idx):
        src, tgt, ref, _ = self._parse_batch(batch)
        _ = ref

        y, aux = self._model_forward(
            src=src,
            ref=None,
            return_aux=True,
        )

        (
            loss,
            mae_loss,
            ssim_score,
            lab_loss,
            param_reg_loss,
            de_loss,
            lab_moment_loss,
            sat_guard_loss,
        ) = self._composite_loss(y, tgt, aux)

        with torch.no_grad():
            y_eval = y.detach()
            tgt_eval = tgt.detach()
            psnr_score = self.psnr_metric(y_eval, tgt_eval)
            sam_score = self.sam_metric(y_eval, tgt_eval)
            de_score = self.de_metric(y_eval[:, self.metrics_channels], tgt_eval[:, self.metrics_channels])

        self.log("mae", mae_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("lab", lab_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("param_reg", param_reg_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("de_loss", de_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("lab_moment", lab_moment_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("sat_guard", sat_guard_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("psnr", psnr_score, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("ssim", ssim_score, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("sam", sam_score, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("de", de_score, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        src, tgt, ref, _ = self._parse_batch(batch)
        _ = ref

        y, aux = self._model_forward(
            src=src,
            ref=None,
            return_aux=True,
        )

        (
            loss,
            mae_loss,
            ssim_score,
            lab_loss,
            param_reg_loss,
            de_loss,
            lab_moment_loss,
            sat_guard_loss,
        ) = self._composite_loss(y, tgt, aux)
        psnr_score = self.psnr_metric(y, tgt)
        sam_score = self.sam_metric(y, tgt)
        de_score = self.de_metric(y[:, self.metrics_channels], tgt[:, self.metrics_channels])

        self.log("val_mae", mae_loss, prog_bar=True, logger=True)
        self.log("val_lab", lab_loss, prog_bar=False, logger=True)
        self.log("val_param_reg", param_reg_loss, prog_bar=False, logger=True)
        self.log("val_de_loss", de_loss, prog_bar=False, logger=True)
        self.log("val_lab_moment", lab_moment_loss, prog_bar=False, logger=True)
        self.log("val_sat_guard", sat_guard_loss, prog_bar=False, logger=True)
        self.log("val_psnr", psnr_score, prog_bar=True, logger=True)
        self.log("val_ssim", ssim_score, prog_bar=True, logger=True)
        self.log("val_sam", sam_score, prog_bar=True, logger=True)
        self.log("val_de", de_score, prog_bar=True, logger=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        src, tgt, ref, _ = self._parse_batch(batch)
        _ = ref

        y, aux = self._model_forward(
            src=src,
            ref=None,
            return_aux=True,
        )

        (
            loss,
            mae_loss,
            ssim_score,
            lab_loss,
            param_reg_loss,
            de_loss,
            lab_moment_loss,
            sat_guard_loss,
        ) = self._composite_loss(y, tgt, aux)
        psnr_score = self.psnr_metric(y, tgt)
        sam_score = self.sam_metric(y, tgt)
        de_score = self.de_metric(y[:, self.metrics_channels], tgt[:, self.metrics_channels])

        self.log("test_mae", mae_loss, prog_bar=True, logger=True)
        self.log("test_lab", lab_loss, prog_bar=False, logger=True)
        self.log("test_param_reg", param_reg_loss, prog_bar=False, logger=True)
        self.log("test_de_loss", de_loss, prog_bar=False, logger=True)
        self.log("test_lab_moment", lab_moment_loss, prog_bar=False, logger=True)
        self.log("test_sat_guard", sat_guard_loss, prog_bar=False, logger=True)
        self.log("test_psnr", psnr_score, prog_bar=True, logger=True)
        self.log("test_ssim", ssim_score, prog_bar=True, logger=True)
        self.log("test_sam", sam_score, prog_bar=True, logger=True)
        self.log("test_de", de_score, prog_bar=True, logger=True)
        self.log("test_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss}

    def on_predict_epoch_start(self) -> None:
        self.sum_mae = 0.0
        self.sum_psnr = 0.0
        self.sum_ssim = 0.0
        self.sum_sam = 0.0
        self.sum_de = 0.0
        self.start_time = 0.0

    def predict_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.start_time = time.perf_counter()

        src, tgt, ref, names = self._parse_batch(batch)
        _ = ref
        y = self._model_forward(
            src=src,
            ref=None,
            return_aux=False,
        )

        elapsed = time.perf_counter() - self.start_time

        mae_score = self.mae_loss(y, tgt)
        psnr_score = self.psnr_metric(y, tgt)
        ssim_score = self.ssim_metric(y, tgt)
        sam_score = self.sam_metric(y, tgt)
        de_score = self.de_metric(y[:, self.metrics_channels], tgt[:, self.metrics_channels])

        self.sum_mae += float(mae_score.item())
        self.sum_psnr += float(psnr_score.item())
        self.sum_ssim += float(ssim_score.item())
        self.sum_sam += float(sam_score.item())
        self.sum_de += float(de_score.item())
        n = 1 + batch_idx

        name = names[0] if names is not None else f"sample_{batch_idx:06d}"
        Logger.info(
            {
                name: {
                    "CUR": {
                        "mae": float(mae_score.item()),
                        "psnr": float(psnr_score.item()),
                        "ssim": float(ssim_score.item()),
                        "sam": float(sam_score.item()),
                        "de": float(de_score.item()),
                    },
                    "AVG": {
                        "mae": self.sum_mae / n,
                        "psnr": self.sum_psnr / n,
                        "ssim": self.sum_ssim / n,
                        "sam": self.sum_sam / n,
                        "de": self.sum_de / n,
                    },
                    "TIME": elapsed / n,
                }
            }
        )

        return {"loss": de_score}
