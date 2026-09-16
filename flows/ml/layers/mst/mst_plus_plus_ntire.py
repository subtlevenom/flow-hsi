"""MST++ adapted to hyperspectral image restoration.

For the NTIRE 31->31 task the model is built with ``in_channels = out_channels = 31``.

Reference: Cai et al., "MST++: Multi-stage Spectral-wise Transformer for Efficient
Spectral Reconstruction" (CVPRW 2022).
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

from ..registry import register


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_.",
            stacklevel=2,
        )
    with torch.no_grad():
        lo = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    denom = {"fan_in": fan_in, "fan_out": fan_out, "fan_avg": (fan_in + fan_out) / 2}[mode]
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class MS_MSA(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """x_in: [b, h, w, c] -> out: [b, h, w, c]."""
        b, h, w, c = x_in.shape
        n = h * w
        x = x_in.reshape(b, n, c)
        q_inp, k_inp, v_inp = self.to_q(x), self.to_k(x), self.to_v(x)

        # [b, n, heads*d] -> [b, heads, n, d]  (replaces einops.rearrange)
        def split_heads(t):
            return t.view(b, n, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        q, k, v = map(split_heads, (q_inp, k_inp, v_inp))
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v                       # [b, heads, d, n]
        x = x.permute(0, 3, 1, 2)          # [b, n, heads, d]
        x = x.reshape(b, n, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([MS_MSA(dim=dim, dim_head=dim_head, heads=heads), PreNorm(dim, FeedForward(dim=dim))])
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        """x: [b, c, h, w] -> [b, c, h, w]."""
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class MST(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=(2, 4, 4)):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        MSAB(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                        nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_stage *= 2

        self.bottleneck = MSAB(dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                        nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                        MSAB(
                            dim=dim_stage // 2,
                            num_blocks=num_blocks[stage - 1 - i],
                            dim_head=dim,
                            heads=(dim_stage // 2) // dim,
                        ),
                    ]
                )
            )
            dim_stage //= 2

        self.mapping = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        fea = self.embedding(x)
        fea_encoder = []
        for msab, downsample in self.encoder_layers:
            fea = msab(fea)
            fea_encoder.append(fea)
            fea = downsample(fea)
        fea = self.bottleneck(fea)
        for i, (upsample, fusion, block) in enumerate(self.decoder_layers):
            fea = upsample(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = block(fea)
        return self.mapping(fea) + x


class MST_Plus_Plus(nn.Module):
    def __init__(self, in_channels=31, out_channels=31, n_feat=31, stage=3):
        super().__init__()
        if n_feat != 31:
            # The internal MST blocks are hard-wired to dim=31; n_feat must match.
            raise ValueError("MST_Plus_Plus requires n_feat == 31")
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=1, bias=False)
        self.body = nn.Sequential(*[MST(dim=31, stage=2, num_blocks=(1, 1, 1)) for _ in range(stage)])
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb = wb = 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h = h + x
        return h[:, :, :h_inp, :w_inp]


@register("model", "mst_plus_plus")
def build(in_channels: int = 31, out_channels: int = 31, n_feat: int = 31, stage: int = 3, **_):
    return MST_Plus_Plus(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, stage=stage)
