import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flows.ml.layers.mst import MSAB

# MSAB из MST: LayerNorm + MS-MSA + FFN
class _MSAB(nn.Module):
    def __init__(self, dim, num_heads=4, ff_dim=None, mask=None):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        self.norm1 = nn.LayerNorm(dim)
        self.mhsa = nn.MultiheadAttention(dim, num_heads, batch_first=True)  # spectral MSA
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, dim)
        )
        self.dropout = nn.Dropout(0.1)
        self.mask = mask  # optional mask из MST

    def forward(self, x):
        # x: [B, C, H, W] -> spectral tokens [B*H*W, C, 1] или [B, C, HW]
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        attn_out = self.mhsa(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))[0]
        x = x_flat + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x.transpose(1, 2).reshape(B, C, H, W)

# MS-MSA: Spectral-wise attention (каналы как tokens)
class SpectralMSA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, HW, C = x.shape
        qkv = self.qkv(x).reshape(B, HW, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, HW, C)
        return self.proj(out)

# 1. Illumination Estimator (без изменений)
class IlluminationEstimator(nn.Module):
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

# 2. Color Context Encoder с MSAB-блоками (как в MST encoder)
class ColorContextEncoderMSAB(nn.Module):
    def __init__(self, in_ch=3, feat_ch=64, num_blocks=4):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, feat_ch, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(feat_ch)

        self.blocks = nn.ModuleList(
            [
                MSAB(
                    dim=feat_ch,
                    dim_head=feat_ch // 4,
                    heads=4,
                    num_blocks=2,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.proj = nn.Conv2d(feat_ch, feat_ch, 1)

    def forward(self, x):
        feat = F.gelu(self.bn(self.embed(x)))  # [B, C, H/2, W/2]

        for block in self.blocks:
            feat = block(feat)  # MSAB stack как в MST

        return self.proj(self.up(feat))

# 3. Parameter Generator с MSAB + Cross-Attention
class SAGFParameterGeneratorMSAB(nn.Module):
    def __init__(self, feat_ch=64, M=8, num_blocks=2):
        super().__init__()
        self.M = M
        # self.cross_attn = nn.MultiheadAttention(feat_ch, 4, batch_first=True)
        self.msab_blocks = nn.ModuleList(
            [
                MSAB(
                    dim=feat_ch,
                    dim_head=feat_ch // 4,
                    heads=4,
                    num_blocks=2,
                )
                for _ in range(num_blocks)
            ]
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(feat_ch * 2, 128, 3, padding=1, bias=False),  # illum + color
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.param_head = nn.Conv2d(128, 3 * M, 1)

    def forward(self, src_illum, src_color, ref_color=None):
        B, C, H, W = src_color.shape

        if ref_color is None:
            ref_color = torch.zeros_like(src_color)

        # Cross-attention: source vs ref (flatten spatial)
        src_flat = src_color.flatten(2).transpose(1, 2)  # [B, HW, C]
        ref_flat = ref_color.flatten(2).transpose(1, 2)
        # cross_out, _ = self.cross_attn(src_flat, ref_flat, ref_flat)
        cross_out = src_flat.transpose(1, 2).reshape(B, C, H, W)

        # MSAB refinement
        feat = cross_out
        for block in self.msab_blocks:
            feat = block(feat)

        fused = self.fusion(torch.cat([src_illum, feat], dim=1))
        params = self.param_head(fused)
        return params

# 4. SAGF Core (формула 3, без изменений)
class SAGFConvCore(nn.Module):
    def __init__(self, ch=64, M=8, q_max=7):
        super().__init__()
        self.M = M
        self.q_max = q_max
        self.out_head = nn.Sequential(
            nn.Conv2d(q_max + 1, ch, 1),
            nn.GELU(),
            nn.Conv2d(ch, 3, 1)
        )

    def forward(self, params):
        B, _, H, W = params.shape
        a = params[:, :self.M]
        mu = F.tanh(params[:, self.M:2*self.M]) + 0.5
        sigma = F.softplus(params[:, 2*self.M:3*self.M]) + 1e-4

        q = torch.arange(self.q_max + 1, device=a.device, dtype=a.dtype).view(1, self.q_max + 1, 1, 1)

        phi = 0.0
        for m in range(self.M):
            am = a[:, m:m+1]
            mum = mu[:, m:m+1]
            sigm = sigma[:, m:m+1]
            phi = phi + am * torch.exp(-((q - mum) ** 2) / (2 * sigm ** 2))

        out = torch.sigmoid(self.out_head(phi))
        return out

# 5. Полная модель с MSAB из MST
class SAGF(nn.Module):
    def __init__(self, M=8, q_max=7):
        super().__init__()
        self.src_illum = IlluminationEstimator(3, 32)
        self.src_color = ColorContextEncoderMSAB(3, 32, num_blocks=4)  # MSAB stack
        self.ref_color = ColorContextEncoderMSAB(3, 32, num_blocks=4)
        
        self.param_gen = SAGFParameterGeneratorMSAB(feat_ch=32, M=M, num_blocks=2)
        self.sagf_core = SAGFConvCore(M=M, q_max=q_max)

    def forward(self, src, ref=None):
        src_illum = self.src_illum(src)
        src_color = self.src_color(src)
        
        if ref is not None:
            ref_color = self.ref_color(ref)
        else:
            ref_color = torch.zeros_like(src_color)
        
        params = self.param_gen(src_illum, src_color, ref_color)
        out = self.sagf_core(params)
        return out
