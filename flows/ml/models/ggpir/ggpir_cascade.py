"""GGPIR-Cascade
Исходник - ggpir_designs.MSTppGGPIRMS
Внешняя зависимость - mst_plus_plus.py

Архитектура
--------------
Реконструкция ГСИ каскадом из трёх идентичных стадий. Каждая стадия - остаточный корректор,
который получает входное ГСИ в одном из масштабов (1/4, 1/2, 1/1) и улучшает апсемпленный
выход предыдущей, более грубой стадии.

Внутри стадии спектр раскладывается на 31 узкое перекрывающееся окно по 3 соседние полосы,
и окна уходят в batch-измерение - так каждое окно идёт своим пространственным путём. Каждое
окно переписывается банком GGP-юнитов: небольшая сеть предсказывает для каждого пикселя 
параметры (a, m, s) набора гауссиан, и эти гауссианы сворачиваются с признаками самого окна 
в один выходной канал. Соседние стадии связаны cross-scale attention: тонкая шкала решает, 
каким полосам грубого предсказания доверять.

Результаты
--------------
Батчсайз на обучении 4, 128x128 патчи, 20 эпох, seed 42, AdamW (lr 2e-4, wd 0.05),
cosine annealing to 1e-6, loss = L1 + 0.15*(1 - SSIM), grad-clip 0.5, AMP.

    | model                          | params |  NTIRE  | ICVL  | CAVE  |
    |--------------------------------|--------|---------|-------|-------|
    | GGPIR-Cascade (этот файл)      | 8.216M | 44.498  | 51.37 | 40.09 |
    | param-matched MST++            | 8.07M  | 44.152  |  TBD  |  TBD  |

PSNR посчитан как в flow-hsi: batch-mean при bs=2, полный кадр.

Память
--------------
Пиковая GPU память, измерена torch.cuda.max_memory_allocated().
Gradient checkpointing ON, иначе батч размера 4 - не влезает.

Обучение, батч 4, 128x128 патчи, AMP (32 GB card):

    | model                                    | params | peak    |
    |------------------------------------------|--------|---------|
    | GGPIR-Cascade (этот файл)                | 8.216M | 15.7 GB |
    | param-matched MST++                      | 8.07M  |  8.2 GB |

Обучение и инференс, измерено на карте 11GB (батч 1 если не указано иное):

    | setting                            | fp32    | AMP     |
    |------------------------------------|---------|---------|
    | инференс, 128x128                  | 0.93 GB | 0.55 GB |
    | инференс, 256x256                  | 3.59 GB | 2.04 GB |
    | инференс, 482x512 (full frame)     | OOM     | 7.53 GB |
    | шаг обучения, батч 1, 128x128      |    -    | 3.74 GB |
    | шаг обучения, батч 2, 128x128      |    -    | 7.37 GB |
    | шаг обучения, батч 4, 128x128      |    -    | OOM     |

Для full-frame инференса на 11GB карте нужен AMP (в fp32 требует ~13.5 GB).
Обучение с батчсайзом 4, как в результатах выше, требует карту 24+ GB.

Веса
--------------
Веса, обученные с mstpp_ggpir_ms (cross_scale_attn=True, все параметры дефолтные),
загружаются в этот класс напрямую:

    model.load_state_dict(torch.load(path, map_location="cpu")["model"])

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ..registry import register
from .mst_plus_plus import MST, MS_MSA

__all__ = ["GGPIRCascade", "build_ggpir_cascade"]


# 1. small blocks
class LayerNorm(nn.Module):
    """LayerNorm по каналам для тензора (B, C, H, W)."""

    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.body(x)
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class FFN(nn.Module):
    """Pointwise - depthwise - pointwise; смешивает выходы GGP-блока."""

    def __init__(self, in_channels: int, hidden_channels: int | None = None,
                 out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.pointwise1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1,
                                   groups=hidden_channels)
        self.pointwise2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise2(self.act(self.depthwise(self.pointwise1(x))))


# 2. GGP core
class ParamGenerator(nn.Module):
    """Генератор MST++: M каналов -> (2M+1) параметров гауссиан одного GGP-юнита.

    MST++ внутри уменьшает разрешение в 2**stage раз, поэтому вход паддится отражением до
    кратного размера и обрезается обратно.
    """

    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 31,
                 gen_blocks: int = 1, mst_stage: int = 2):
        super().__init__()
        self.mst_stage = mst_stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, 3, padding=1, bias=False)
        self.body = nn.Sequential(*[MST(dim=n_feat, stage=mst_stage, num_blocks=(1, 1, 1))
                                    for _ in range(gen_blocks)])
        self.conv_out = nn.Conv2d(n_feat, out_channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h_inp, w_inp = x.shape
        m = 1 << self.mst_stage
        pad_h = (m - h_inp % m) % m
        pad_w = (m - w_inp % m) % m
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        x = self.conv_out(self.body(self.conv_in(x)))
        return x[:, :, :h_inp, :w_inp]


class GaussianCollapse(nn.Module):
    """Коллапс M каналов в 1 суммой гауссиан. Параметры (a, m, s) предсказываются на каждый пиксель.

    Считается в fp32 с клампом экспоненты: под AMP выученный масштаб s может переполнить fp16
    в (x - m) * s, и дальше inf * 0 даёт NaN в градиентах.
    """

    def forward(self, x: torch.Tensor, a: torch.Tensor, m: torch.Tensor,
                s: torch.Tensor) -> torch.Tensor:
        z = (x.float() - m.float()) * s.float()
        y = a.float() * torch.exp(-0.5 * (z * z).clamp(max=50.0))
        return y.sum(dim=1, keepdim=True)


class GGPUnit(nn.Module):
    """Один GGP-юнит: предсказывает (a, m, s) по M каналам и коллапсирует эти каналы в один.

    Генератор лежит в поле self.encoder - имя сохранено ради совместимости с весами; это
    генератор параметров, а не проектор полос ниже.
    """

    def __init__(self, in_channels: int, n_feat: int = 31, gen_blocks: int = 1,
                 mst_stage: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = ParamGenerator(in_channels, 2 * in_channels + 1, n_feat=n_feat,
                                      gen_blocks=gen_blocks, mst_stage=mst_stage)
        self.gaussian = GaussianCollapse()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        w = self.encoder(x)
        a, m, s = w[:, :1], w[:, 1:c + 1], w[:, c + 1:]
        return self.gaussian(x, a, m, s)


class GGPUnitBank(nn.Module):
    """n_ggp независимых GGP-юнитов на одном входе, выходы конкатенируются в n_ggp каналов.

    В каждом юните полный генератор MST++, их активации и составляют основную часть памяти,
    поэтому на обучении каждый юнит оборачивается в gradient checkpoint.
    """

    def __init__(self, in_channels: int, n_ggp: int, grad_checkpoint: bool = True, **gen_kw):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.layers = nn.ModuleList([GGPUnit(in_channels, **gen_kw) for _ in range(n_ggp)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and x.requires_grad:
            outs = [checkpoint(unit, x, use_reentrant=False) for unit in self.layers]
        else:
            outs = [unit(x) for unit in self.layers]
        return torch.cat(outs, dim=1)


class GGPBlock(nn.Module):
    """concat[x, LayerNorm(x), банк GGP] -> FFN -> M каналов."""

    def __init__(self, in_channels: int = 3, n_ggp: int = 7, grad_checkpoint: bool = True,
                 **gen_kw):
        super().__init__()
        self.layer = GGPUnitBank(in_channels, n_ggp, grad_checkpoint=grad_checkpoint, **gen_kw)
        self.norm = LayerNorm(in_channels)
        self.ffn = FFN(in_channels=n_ggp + 2 * in_channels, out_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        y = self.layer(x)
        return self.ffn(torch.cat([x, z, y], dim=1))


# 3. the band fold
class BandProjector(nn.Module):
    """Спектральные окна в батче: (B, C, H, W) -> (B*C, M, H, W).

    cat([x] * M) оборачивает ось полос M раз, дальше rearrange режет её на C групп по M.
    Слот n несёт полосы (M*n + k) mod C, k = 0..M-1 - скользящее окно из M соседних полос.
    Старты M*n mod C пробегают все полосы, так что C слотов - это полный набор скользящих
    окон ширины M по кругу, только в переставленном порядке; каждая полоса попадает в M окон.
    Групповая свёртка (groups=C) группирует те же тройки: гейт применяется к окну целиком.
    """

    def __init__(self, in_channels: int = 31, out_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.SiLU(inplace=True)
        all_channels = in_channels * out_channels
        self.weights = nn.Conv2d(all_channels, all_channels, 3, padding=1, groups=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x] * self.out_channels, dim=1)
        x = x * (1.0 + self.act(self.weights(x)))
        return rearrange(x, "b (n c) h w -> (b n) c h w",
                         n=self.in_channels, c=self.out_channels)


class BandAggregator(nn.Module):
    """Окна обратно из батча с усреднением: (B*C, M, H, W) -> (B, C, H, W).

    Зеркало BandProjector: выходная полоса j - среднее слотов (i*C + j) // M по i = 0..M-1,
    то есть тех M окон, в которые она входила. При C=31, M=3 полоса 0 - среднее слотов 0, 10, 20.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 31):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.SiLU(inplace=True)
        all_channels = in_channels * out_channels
        self.weights = nn.Conv2d(all_channels, all_channels, 3, padding=1, groups=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "(b n) c h w -> b (n c) h w",
                      n=self.out_channels, c=self.in_channels)
        x = x * (1.0 + self.act(self.weights(x)))
        groups = [x[:, i * self.out_channels:(i + 1) * self.out_channels]
                  for i in range(self.in_channels)]
        return sum(groups) / float(self.in_channels)


# 4. cascade stage
class CascadeStage(nn.Module):
    """Одна остаточная стадия: проекция -> GGP-блок -> агрегация, плюс глобальный skip-connection."""

    def __init__(self, channels: int = 31, proj_channels: int = 3, n_ggp: int = 7,
                 gen_blocks: int = 1, gen_stage: int = 2, grad_checkpoint: bool = True):
        super().__init__()
        self.encoder = BandProjector(channels, proj_channels)
        self.layer = GGPBlock(proj_channels, n_ggp, grad_checkpoint=grad_checkpoint,
                              n_feat=31, gen_blocks=gen_blocks, mst_stage=gen_stage)
        self.decoder = BandAggregator(proj_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.decoder(self.layer(self.encoder(x)))


# 5. cross-scale attention
class CrossScaleAttention(nn.Module):
    """Связь двух соседних шкал каскада вместо простого сложения xs + up(y_coarse).

    Обе шкалы конкатенируются по оси полос (2C полос) и проходят через спектральный
    self-attention из MST++, потом 1x1 свёртка проецирует поправку обратно в C полос.
    """

    def __init__(self, channels: int = 31):
        super().__init__()
        self.attn = MS_MSA(dim=2 * channels, dim_head=2 * channels, heads=1)
        self.proj = nn.Conv2d(2 * channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, xs: torch.Tensor, up_prev: torch.Tensor) -> torch.Tensor:
        z = torch.cat([xs, up_prev], dim=1).permute(0, 2, 3, 1)   # (B, H, W, 2C)
        a = self.attn(z).permute(0, 3, 1, 2)                      # (B, 2C, H, W)
        return xs + up_prev + self.proj(a)


# 6. the model
class GGPIRCascade(nn.Module):
    """Каскад GGP-стадий от грубой шкалы к тонкой с cross-scale attention.

    При дефолтной геометрии пирамида входов - (H/4, W/4), (H/2, W/2), (H, W), число GGP-юнитов
    по стадиям 3 / 5 / 7. Если x_s - вход, отресайзенный в шкалу s, то

        y_0 = stage_0(x_0)
        y_s = stage_s( fuse( x_s, upsample(y_{s-1}) ) )      для s = 1, 2
        out = y_2

    где fuse - это CrossScaleAttention при cross_scale_attn=True (дефолт, именно эта
    конфигурация в таблице результатов) и обычная сумма иначе.

    Минимальный размер входа - 12 пикселей по каждой стороне: самая грубая шкала это H/4, а
    генератор паддит отражением до кратного 2**gen_stage, и паддинг должен быть меньше входа.
    Важно, если нарезать большую сцену на тайлы.

    Args:
        in_channels/out_channels: число спектральных полос, должны совпадать.
        proj_channels: M, ширина пути одного спектрального окна.
        n_ggp_per_scale: число GGP-юнитов по стадиям, от грубой к тонкой; длина задаёт число шкал.
        gen_blocks/gen_stage: глубина генератора параметров и глубина его даунсемплинга.
        grad_checkpoint: пересчитывать каждый GGP-юнит на обратном проходе; примерно +30% времени.
        cross_scale_attn: attention-связь между шкалами вместо суммы.
    """

    def __init__(self, in_channels: int = 31, out_channels: int = 31, proj_channels: int = 3,
                 n_ggp_per_scale: tuple[int, ...] = (3, 5, 7), gen_blocks: int = 1,
                 gen_stage: int = 2, grad_checkpoint: bool = True,
                 cross_scale_attn: bool = True):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("GGPIRCascade is residual end-to-end; it needs in_channels == "
                             f"out_channels (got {in_channels} and {out_channels})")
        if len(n_ggp_per_scale) < 1:
            raise ValueError("n_ggp_per_scale must name at least one scale")
        self.n_scales = len(n_ggp_per_scale)
        self.stages = nn.ModuleList(
            [CascadeStage(in_channels, proj_channels, n_ggp=g, gen_blocks=gen_blocks,
                          gen_stage=gen_stage, grad_checkpoint=grad_checkpoint)
             for g in n_ggp_per_scale]          # грубая -> тонкая
        )
        self.fuse = nn.ModuleList(
            [CrossScaleAttention(in_channels) for _ in range(self.n_scales - 1)]
        ) if cross_scale_attn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        sizes = [(max(1, h >> s), max(1, w >> s)) for s in range(self.n_scales - 1, -1, -1)]
        y = None
        for i, (stage, size) in enumerate(zip(self.stages, sizes)):
            xs = x if size == (h, w) else F.interpolate(x, size=size, mode="bilinear",
                                                        align_corners=False)
            if y is not None:
                up = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
                xs = self.fuse[i - 1](xs, up) if self.fuse is not None else xs + up
            y = stage(xs)
        return y


@register("model", "ggpir_cascade")
def build_ggpir_cascade(in_channels: int = 31, out_channels: int = 31, proj_channels: int = 3,
                        n_ggp_per_scale=(3, 5, 7), gen_blocks: int = 1, gen_stage: int = 2,
                        grad_checkpoint: bool = True, cross_scale_attn: bool = True, **_):
    return GGPIRCascade(in_channels, out_channels, proj_channels, tuple(n_ggp_per_scale),
                        gen_blocks, gen_stage, grad_checkpoint, cross_scale_attn)
