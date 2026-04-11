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
from flows.tools.utils import models, text
from flows.core import Logger
from flows.ml.losses import GPDFLoss
from ..models import Flow
from ..metrics import PSNR, SSIM, SAM, DeltaE
from flows.ml.layers.sep_gpd import MultivariateNormal


class USGSPipeline(L.LightningModule):

    def __init__(
        self,
        model: Flow,
        optimizer: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_reg: float = 1e-4,
        warmup_epochs: int = 10,
        metrics_channels: List[int] = [0, 1, 2],
    ) -> None:
        super(USGSPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.ggpd_loss = GPDFLoss()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.mae_loss = nn.L1Loss(reduction="mean")
        self.de_metric = DeltaE()
        self.sam_metric = SAM()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.metrics_channels = metrics_channels
        self.lambda_reg = lambda_reg
        self.warmup_epochs = warmup_epochs

        self.save_hyperparameters(ignore=["model"])

    def _initialize_model_weights(self):
        """Инициализация под HC-USGS с AdvancedParameterEncoder."""
        usgs = self.model.layers.usgs
        with torch.no_grad():
            M = usgs.M
            Q = usgs.Q

            # 1. Инициализация энкодеров параметров (AdvancedParameterEncoder)
            # В новой модели финальный слой - это fusion[-1]
            encoders = [usgs.enc_amplitude, usgs.enc_center, usgs.enc_sigma]
            for enc in encoders:
                nn.init.constant_(enc.fusion[-1].weight, 0.0)
                nn.init.constant_(enc.fusion[-1].bias, 0.0)

            # 2. Специфические смещения для центров (mu) и сигм
            # Центры распределяем равномерно [0.1, 0.9]
            initial_mu = torch.linspace(0.1, 0.9, M)
            usgs.enc_center.fusion[-1].bias.copy_(initial_mu)

            # Сигмы делаем широкими на старте
            usgs.enc_sigma.fusion[-1].bias.fill_(1.0)

            # 3. Генератор узлов (node_generator)
            # Linear слой перед Sigmoid имеет индекс -2
            node_linear = usgs.node_generator[-2]
            nn.init.constant_(node_linear.weight, 0.0)
            node_bias = torch.linspace(-3.0, 3.0, Q)
            node_linear.bias.copy_(node_bias)

            # 4. Блок суперпозиции (chi_superposition)
            # Финальный слой Conv2d (индекс -1) в 0
            nn.init.constant_(usgs.chi_superposition[-1].bias, 0.0)
            nn.init.normal_(usgs.chi_superposition[-1].weight, std=1e-4)

            # Первый слой (групповая свертка)
            nn.init.kaiming_normal_(usgs.chi_superposition[0].weight, mode='fan_out', nonlinearity='relu')

    def setup(self, stage: str) -> None:
        """
        Initialize model weights before training
        """
        if stage == "fit" or stage is None:
            # Общая инициализация Kaiming для всех слоев
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            # Специфическая инициализация для USGS
            self._initialize_model_weights()

        # MODEL_PATH = '.experiments/ggpd.huawei/logs/checkpoints/_last.ckpt'
        # models.load_model(self.model.layers.encoder, 'model.layers.encoder', MODEL_PATH)
        # models.require_grad(self.model.layers.encoder.gpd_x, requires_grad=False)

        Logger.info("Initialized model weights with isp pipeline.")

    def configure_optimizers(self):
        if self.optimizer_type == "adamw":
            usgs_params = []
            base_params = []
            for name, param in self.model.named_parameters():
                # Включаем node_generator и новые энкодеры в группу с повышенным LR
                if any(x in name for x in ["enc_", "chi", "node_generator"]):
                    usgs_params.append(param)
                else:
                    base_params.append(param)

            optimizer = optim.AdamW([{
                "params": base_params,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            }, {
                "params": usgs_params,
                "lr": self.lr * 2,
                "weight_decay": 0.0,
            }])

            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    self.warmup_epochs,
                    self.warmup_epochs + 40,
                    self.warmup_epochs + 80,
                ],
                gamma=0.5)

        elif self.optimizer_type == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=500,
                T_mult=1,
                eta_min=1e-5,
            )
        elif self.optimizer_type == "sgd":
            optimizer = optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=500,
                T_mult=1,
                eta_min=1e-5,
            )
        else:
            raise ValueError(f"unsupported optimizer_type: {self.optimizer_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_start(self):
        usgs = self.model.layers.usgs
        # Warmup: обучаем только базовую освещенность (illum_estimator) и stem
        is_warmup = self.current_epoch < self.warmup_epochs
        for name, param in usgs.named_parameters():
            if any(x in name for x in ["enc_", "chi", "node_generator"]):
                param.requires_grad = not is_warmup
            else:
                param.requires_grad = True

    def _spatial_smoothness_loss(self, x):
        """TV-loss для тензоров произвольной размерности [..., H, W]."""
        # Берем разности по последним двум осям (H, W)
        dy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        dx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        return torch.mean(dy) + torch.mean(dx)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pred = self.model(src=x, tgt=y)
        return pred["res"]

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # Финальный результат модели
        y, a_maps, mu_maps, sigma_maps = self.model.layers.usgs(src, True)

        # 1. Комбинированный лосс (L1 + SSIM) - критично для cmKAN
        # Основные лоссы
        mae_loss = self.mae_loss(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        loss_ssim = 1 - ssim_loss

        # 2. Регуляризация (TV-loss для гладкости карт параметров)
        reg_smooth = (
            self._spatial_smoothness_loss(a_maps)
            + self._spatial_smoothness_loss(mu_maps)
            + self._spatial_smoothness_loss(sigma_maps)
        )

        # 3. Repulsion Loss для узлов (чтобы не слипались)
        # Извлекаем узлы напрямую из модели для лосса
        q_nodes = self.model.layers.usgs.node_generator(self.model.layers.usgs.stem(self.model.layers.usgs.coord_adder(src)))
        q_nodes, _ = torch.sort(q_nodes, dim=1)
        node_dist = torch.diff(q_nodes, dim=1)
        loss_repulsion = torch.mean(F.relu(0.1 - node_dist)) # Минимальный зазор 0.1

        loss = 0.9 * mae_loss + 0.15 * loss_ssim + self.lambda_reg * reg_smooth + 0.01 * loss_repulsion

        psnr_loss = self.psnr_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        de_loss = self.de_metric(
            y[:, self.metrics_channels], tgt[:, self.metrics_channels]
        )

        self.log("mae", mae_loss, prog_bar=True, logger=True)
        self.log("psnr", psnr_loss, prog_bar=True, logger=True)
        self.log("ssim", ssim_loss, prog_bar=True, logger=True)
        self.log("sam", sam_loss, prog_bar=True, logger=True)
        self.log("de", de_loss, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src)

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        de_loss = self.de_metric(
            y[:, self.metrics_channels], tgt[:, self.metrics_channels]
        )
        loss = mae_loss

        self.log("val_mae", mae_loss, prog_bar=True, logger=True)
        self.log("val_psnr", psnr_loss, prog_bar=True, logger=True)
        self.log("val_ssim", ssim_loss, prog_bar=True, logger=True)
        self.log("val_sam", sam_loss, prog_bar=True, logger=True)
        self.log("val_de", de_loss, prog_bar=True, logger=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        src, tgt = batch

        y = self(src)

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        de_loss = self.de_metric(
            y[:, self.metrics_channels], tgt[:, self.metrics_channels]
        )
        loss = mae_loss

        self.log("test_mae", mae_loss, prog_bar=True, logger=True)
        self.log("test_psnr", psnr_loss, prog_bar=True, logger=True)
        self.log("test_ssim", ssim_loss, prog_bar=True, logger=True)
        self.log("test_sam", sam_loss, prog_bar=True, logger=True)
        self.log("test_de", de_loss, prog_bar=True, logger=True)
        self.log("test_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss}

    sum_mae = 0
    sum_psnr = 0
    sum_ssim = 0
    sum_sam = 0
    sum_de = 0
    start_time = 0

    def predict_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.start_time = time.perf_counter()

        src, tgt, name = batch
        y = self(src)
        elapsed = time.perf_counter() - self.start_time

        mae_loss = self.mae_loss(y, tgt)
        psnr_loss = self.psnr_metric(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        de_loss = self.de_metric(
            y[:, self.metrics_channels], tgt[:, self.metrics_channels]
        )

        self.sum_mae += mae_loss
        self.sum_psnr += psnr_loss
        self.sum_ssim += ssim_loss
        self.sum_sam += sam_loss
        self.sum_de += de_loss
        n = 1 + batch_idx

        text.print_json(
            {
                name[0]: {
                    "CUR": {
                        "mae": mae_loss.item(),
                        "psnr": psnr_loss.item(),
                        "ssim": ssim_loss.item(),
                        "sam": sam_loss.item(),
                        "de": de_loss.item(),
                    },
                    "AVG": {
                        "mae": self.sum_mae.item() / n,
                        "psnr": self.sum_psnr.item() / n,
                        "ssim": self.sum_ssim.item() / n,
                        "sam": self.sum_sam.item() / n,
                        "de": self.sum_de.item() / n,
                    },
                    "TIME": elapsed / n,
                },
            }
        )

        return {"loss": de_loss}
