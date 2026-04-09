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
        weight_decay: float = 0,
        lambda_reg: float = 1e-5,
        warmup_epochs: int = 5,
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
        """Инициализация под адаптивную USGS-схему с динамическими узлами."""
        usgs = self.model.layers.usgs
        with torch.no_grad():
            M = usgs.M
            Q = usgs.Q

            # 1. Центры Гауссиан (mu)
            initial_mu = torch.linspace(0.1, 0.9, M)
            usgs.enc_center.up[-1].bias.copy_(initial_mu)
            usgs.enc_center.up[-1].weight.fill_(0)

            # 2. Амплитуды (a)
            usgs.enc_amplitude.up[-1].bias.fill_(0.0)
            usgs.enc_amplitude.up[-1].weight.fill_(0)

            # 3. Сигмы (sigma)
            usgs.enc_sigma.up[-1].bias.fill_(1.0)
            usgs.enc_sigma.up[-1].weight.fill_(0)

            # 4. Генератор узлов (node_generator)
            # ИСПРАВЛЕНО: Обращаемся к Linear слою перед Sigmoid (индекс -2)
            node_linear = usgs.node_generator[-2] 
            nn.init.constant_(node_linear.weight, 0.0)
            # Создаем линейную прогрессию для bias, чтобы узлы были распределены [0.12, 0.88]
            node_bias = torch.linspace(-2.0, 2.0, Q) 
            node_linear.bias.copy_(node_bias)

            # 5. Блок суперпозиции (chi_superposition)
            # ИСПРАВЛЕНО: Последний слой - это Conv2d (индекс -1)
            chi_final = usgs.chi_superposition[-1]
            nn.init.constant_(chi_final.bias, 0.0)
            nn.init.normal_(chi_final.weight, std=1e-4)
            
            # Первый слой chi_superposition (групповая свертка, индекс 0)
            chi_first = usgs.chi_superposition[0]
            nn.init.kaiming_normal_(chi_first.weight, mode='fan_out', nonlinearity='relu')

    def setup(self, stage: str) -> None:
        """
        Initialize model weights before training
        """
        if stage == "fit" or stage is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            self._initialize_model_weights()

        # MODEL_PATH = '.experiments/ggpd.huawei/logs/checkpoints/_last.ckpt'
        # models.load_model(self.model.layers.encoder, 'model.layers.encoder', MODEL_PATH)
        # models.require_grad(self.model.layers.encoder.gpd_x, requires_grad=False)

        Logger.info("Initialized model weights with isp pipeline.")

    def configure_optimizers(self):
        if self.optimizer_type == "adamw":
            # Разделяем параметры для разного LR и Weight Decay
            usgs_params = []
            base_params = []

            for name, param in self.model.named_parameters():
                if "enc_" in name or "chi" in name or "node_generator" in name:
                    usgs_params.append(param)
                else:
                    base_params.append(param)

            optimizer = optim.AdamW(
                [
                    {
                        "params": base_params,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": usgs_params,
                        "lr": self.lr * 5,  # Ускоряем обучение цветовых кривых
                        "weight_decay": 0.0,  # Не зануляем амплитуды принудительно
                    },
                ]
            )

            # Планировщик с учетом выхода из warmup
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    self.warmup_epochs,
                    self.warmup_epochs + 40,
                    self.warmup_epochs + 80,
                ],
                gamma=0.5,
            )
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
        """Разморозка USGS после warmup."""
        usgs = self.model.layers.usgs
        if self.current_epoch < self.warmup_epochs:
            for name, param in usgs.named_parameters():
                if "enc_" in name or "chi" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for param in usgs.parameters():
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
        l1_loss = self.mae_loss(y, tgt)
        ssim_loss = self.ssim_metric(y, tgt)
        loss_ssim = 1 - ssim_loss

        # Основные лоссы
        l1_loss = self.mae_loss(y, tgt)
        reg_smooth = (
            self._spatial_smoothness_loss(a_maps)
            + self._spatial_smoothness_loss(mu_maps)
            + self._spatial_smoothness_loss(sigma_maps)
        )

        # Итоговый лосс: 70% L1 + 30% SSIM + TV-reg
        total_loss = 0.7 * l1_loss + 0.3 * loss_ssim + self.lambda_reg * reg_smooth

        # Регуляризация
        # reg = self.gaussian_regularization(a, mu, sigma)
        # laplace = self.laplacian_pyramid_loss(y, tgt)
        # total_loss = l1 + 0.5 * laplace + self.lambda_reg * reg

        psnr_loss = self.psnr_metric(y, tgt)
        sam_loss = self.sam_metric(y, tgt)
        de_loss = self.de_metric(
            y[:, self.metrics_channels], tgt[:, self.metrics_channels]
        )
        loss = total_loss

        self.log("mae", total_loss, prog_bar=True, logger=True)
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
