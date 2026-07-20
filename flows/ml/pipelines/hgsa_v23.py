import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L

from ..metrics import PSNR, SSIM, DeltaE
# Функции потерь переиспользуются из v22 (это nn.Module, а не LightningModule,
# поэтому их импорт безопасен для селектора пайплайнов, который ищет
# единственный подкласс LightningModule в модуле).
from .hgsa_v22 import (
    LogCoshLoss,
    GradLoss,
    CIELabLoss,
    FrequencyLoss,
    MongeKantorovichLoss,
    DeltaE2000Loss,
)

# ─────────────────────────────────────────────────────────────────────
# PIPELINE v23 (Capacity + Multi-Scale Encoder + Transfer Learning)
# ─────────────────────────────────────────────────────────────────────

class GEOTPipeline_v23(L.LightningModule):
    """Пайплайн v23 с поддержкой переноса обучения (pretrain → fine-tune).

    Реализует пункт плана #13. Логика функций потерь идентична v22, но
    добавлены три возможности дообучения:

    * ``pretrained_ckpt`` — путь к чекпойнту предобучения; на этапе ``fit``
      совместимые по форме веса загружаются в модель (``strict=False``), что
      позволяет переносить знания между датасетами даже при частичном
      несовпадении архитектуры.
    * поэтапная заморозка — при ``finetune=True`` тяжёлое транспортное ядро
      (энкодер + EOT/USGS блоки) замораживается на ``freeze_epochs`` эпох,
      чтобы сначала адаптировать датасет-специфичные ветви (кондиционер и
      финальное слияние), затем размораживается для полного дообучения.
    * пониженный LR дообучения — ``finetune_lr_scale`` масштабирует все
      групповые learning rate при ``finetune=True``.
    """

    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-3,
                 warmup_epochs: int = 50,
                 weight_decay: float = 1e-4,
                 pretrained_ckpt: str = None,
                 finetune: bool = False,
                 freeze_epochs: int = 0,
                 finetune_lr_scale: float = 0.3) -> None:
        super().__init__()
        self._model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Перенос обучения (#13)
        self.pretrained_ckpt = pretrained_ckpt
        self.finetune = finetune
        self.freeze_epochs = freeze_epochs
        self.finetune_lr_scale = finetune_lr_scale
        self._frozen = False

        # Metrics
        self.psnr_metric = PSNR(data_range=1.0)
        self.ssim_metric = SSIM(data_range=1.0)
        self.de_metric = DeltaE()

        # Losses
        self.mae_loss = nn.L1Loss()
        self.lab_loss = CIELabLoss(chroma_weight=1.5)
        self.freq_loss = FrequencyLoss(loss_weight=0.5)
        self.grad_loss = GradLoss()
        self.mk_loss = MongeKantorovichLoss()
        self.de_loss = DeltaE2000Loss()
        self.ssim_loss = SSIM(data_range=1.0)

        # Weights (v23, наследует настройку v22)
        self.w_lab = 3.0
        self.w_freq = 1.2
        self.w_mk = 0.08
        self.w_ssim = 0.25
        self.w_de = 0.5
        self.w_aux = 0.6
        self.w_grad = 0.5

        self.save_hyperparameters(ignore=['model'])

    @property
    def model(self):
        return self._model.layers.hgsa

    # ── Перенос обучения ────────────────────────────────────────────

    def _load_pretrained_weights(self, path: str) -> None:
        """Загружает совместимые по форме веса из чекпойнта предобучения."""
        if not path or not os.path.isfile(path):
            print(f'[v23] pretrained checkpoint not found, training from scratch: {path}')
            return

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

        model_sd = self.model.state_dict()
        prefixes = ('_model.layers.hgsa.', 'model.layers.hgsa.',
                    'layers.hgsa.', 'hgsa.', '_model.', 'model.')
        remapped = {}
        for k, v in sd.items():
            key = k
            for pref in prefixes:
                if key.startswith(pref):
                    key = key[len(pref):]
                    break
            if key in model_sd and model_sd[key].shape == v.shape:
                remapped[key] = v

        self.model.load_state_dict(remapped, strict=False)
        print(f'[v23] loaded {len(remapped)}/{len(model_sd)} pretrained tensors from {path}')

    def _set_transport_frozen(self, frozen: bool) -> None:
        """Замораживает/размораживает перенесённое транспортное ядро."""
        for name, param in self.model.named_parameters():
            if name.startswith('encoder') or name.startswith('eot_usgs'):
                param.requires_grad = not frozen
        self._frozen = frozen

    def setup(self, stage: str = None) -> None:
        if stage != 'fit':
            return

        # При дообучении инициализация «с нуля» не выполняется — веса берутся
        # из предобученного чекпойнта. Иначе инициализируем как в v22.
        if not (self.finetune and self.pretrained_ckpt):
            self._init_from_scratch()

        if self.pretrained_ckpt:
            self._load_pretrained_weights(self.pretrained_ckpt)

        if self.finetune and self.freeze_epochs > 0:
            self._set_transport_frozen(True)

    def _init_from_scratch(self) -> None:
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'hyper' in name or 'phi_net' in name or 'base' in name or 'spline' in name:
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                if 'film' in name:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None: nn.init.zeros_(m.bias)

    def on_train_epoch_start(self) -> None:
        # Разморозка транспортного ядра после стадии заморозки при дообучении.
        if self._frozen and self.current_epoch >= self.freeze_epochs:
            self._set_transport_frozen(False)
            print(f'[v23] unfroze transport core at epoch {self.current_epoch}')

    def configure_optimizers(self):
        # Важно: включаем ВСЕ параметры (в т.ч. временно замороженные при
        # поэтапной заморозке), чтобы после разморозки они уже присутствовали
        # в оптимизаторе. Замороженные параметры не обновляются, т.к. их
        # grad остаётся None.
        groups = {'cond': [], 'enc': [], 'eot': [], 'fuse': []}
        for name, param in self.model.named_parameters():
            if 'conditioner' in name: groups['cond'].append(param)
            elif 'encoder' in name: groups['enc'].append(param)
            elif 'eot_usgs' in name: groups['eot'].append(param)
            else: groups['fuse'].append(param)

        s = self.finetune_lr_scale if self.finetune else 1.0
        lrs = [self.lr * 0.5 * s, self.lr * 0.8 * s, self.lr * 1.8 * s, self.lr * 1.0 * s]

        optimizer = optim.AdamW([
            {'params': groups['cond'], 'lr': lrs[0]},
            {'params': groups['enc'], 'lr': lrs[1]},
            {'params': groups['eot'], 'lr': lrs[2], 'weight_decay': 1e-3},
            {'params': groups['fuse'], 'lr': lrs[3]},
        ], weight_decay=self.weight_decay)

        steps = self.trainer.estimated_stepping_batches
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lrs,
            total_steps=steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def forward(self, x):
        res = self.model(x)['res']
        if self.training: return res  # (final_out, transported_x)
        return torch.clamp(res[0] if isinstance(res, tuple) else res, 0, 1)

    def _compute_loss(self, src, main_out, aux_out, tgt, is_warmup):
        loss_mae = self.mae_loss(main_out, tgt)
        loss_aux = self.mae_loss(aux_out, tgt)

        if is_warmup:
            return loss_mae + self.w_aux * loss_aux, {'mae': loss_mae, 'aux': loss_aux}

        m_c, a_c = torch.clamp(main_out, 0, 1), torch.clamp(aux_out, 0, 1)

        loss_lab = self.lab_loss(m_c, tgt)
        loss_freq = self.freq_loss(m_c, tgt)
        loss_grad = self.grad_loss(m_c, tgt)
        loss_mk = self.mk_loss(src, a_c)
        loss_de = self.de_loss(m_c, tgt)
        loss_ssim = 1.0 - self.ssim_loss(m_c, tgt)

        total = (loss_mae + self.w_lab * loss_lab + self.w_freq * loss_freq +
                 self.w_grad * loss_grad + self.w_aux * loss_aux + self.w_mk * loss_mk +
                 self.w_de * loss_de + self.w_ssim * loss_ssim)

        return total, {'mae': loss_mae, 'lab': loss_lab, 'mk': loss_mk,
                       'de': loss_de, 'ssim': loss_ssim}

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        main_out, aux_out = self(src)
        loss, details = self._compute_loss(src, main_out, aux_out, tgt, self.current_epoch < self.warmup_epochs)
        self.log('train_loss', loss, prog_bar=True)
        for k, v in details.items(): self.log(f'train/{k}', v)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)

        loss_mae = F.l1_loss(y, tgt)
        psnr = self.psnr_metric(y, tgt)
        ssim = self.ssim_metric(y, tgt)
        de = self.de_metric(y, tgt).mean()

        self.log('val_mae', loss_mae, prog_bar=True, sync_dist=True)
        self.log('val_psnr', psnr, prog_bar=True)
        self.log('val_ssim', ssim, prog_bar=True)
        self.log('val_de', de, prog_bar=True)

        return loss_mae

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        y = self(src)

        loss_mae = F.l1_loss(y, tgt)
        psnr = self.psnr_metric(y, tgt)
        ssim = self.ssim_metric(y, tgt)
        de = self.de_metric(y, tgt).mean()

        self.log('test_mae', loss_mae, prog_bar=True, sync_dist=True)
        self.log('test_psnr', psnr, prog_bar=True)
        self.log('test_ssim', ssim, prog_bar=True)
        self.log('test_de', de, prog_bar=True)

        return loss_mae
