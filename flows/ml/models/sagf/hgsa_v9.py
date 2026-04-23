import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# --- Вспомогательные блоки ---


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class GELU(nn.Module):

    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# --- MSAB Компоненты ---


class MaskGuidedMechanism(nn.Module):

    def __init__(self, n_feat, in_channels=3):
        super(MaskGuidedMechanism, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.proj_mask = nn.Conv2d(in_channels,
                                   n_feat,
                                   kernel_size=1,
                                   bias=True)
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat,
                                    n_feat,
                                    kernel_size=5,
                                    padding=2,
                                    bias=True,
                                    groups=n_feat)

    def forward(self, mask):
        if mask.shape[
                1] == self.in_channels and self.in_channels != self.n_feat:
            mask = self.proj_mask(mask)
        mask = self.conv1(mask)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask)))
        return mask * attn_map + mask


class MS_MSA_Context(nn.Module):

    def __init__(self, dim, dim_head, heads, context_dim=0, mask_channels=3):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.input_dim = dim + context_dim

        self.to_qkv = nn.Linear(self.input_dim,
                                dim_head * heads * 3,
                                bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
        )
        # Маска должна генерировать столько же признаков, сколько в сумме у всех голов (dim_head * heads)
        self.mm = MaskGuidedMechanism(dim_head * heads,
                                      in_channels=mask_channels)

    def forward(self, x_in, mask, context_vec=None):
        b, h, w, c = x_in.shape

        if context_vec is not None:
            ctx = context_vec.expand(-1, -1, h, w).permute(0, 2, 3, 1)
            x_for_qkv = torch.cat([x_in, ctx], dim=-1)
        else:
            x_for_qkv = x_in

        x_flat = x_for_qkv.reshape(b, h * w, -1)
        qkv = self.to_qkv(x_flat).chunk(3, dim=-1)

        # q, k, v: [b, heads, n, dim_head]
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            qkv)

        # ИСПРАВЛЕНИЕ ТУТ:
        # 1. Получаем маску [b, (heads*dim_head), h, w]
        mask_features = self.mm(mask)
        # 2. Переставляем в [b, h*w, (heads*dim_head)]
        mask_features = mask_features.permute(0, 2, 3, 1).reshape(b, h * w, -1)
        # 3. Нарезаем на головы [b, heads, n, dim_head]
        mask_attn = rearrange(mask_features,
                              'b n (h d) -> b h n d',
                              h=self.num_heads)

        # Теперь размерности v и mask_attn совпадают: [b, heads, n, dim_head]
        v = v * mask_attn

        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 3, 1, 2).reshape(b, h * w, -1)
        out_c = self.proj(out).view(b, h, w, c)

        # Positional embedding
        v_res = rearrange(v, 'b h d n -> b (h d) n').reshape(
            b, self.num_heads * self.dim_head, h, w)
        out_p = self.pos_emb(v_res).permute(0, 2, 3, 1)

        return out_c + out_p


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult,
                      dim * mult,
                      3,
                      1,
                      1,
                      bias=False,
                      groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MSAB_Context(nn.Module):

    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 num_blocks,
                 context_dim=0,
                 mask_channels=3):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList([
                    MS_MSA_Context(dim, dim_head, heads, context_dim,
                                   mask_channels),
                    PreNorm(dim, FeedForward(dim))
                ]))

    def forward(self, x, mask, context_list=None):
        x = x.permute(0, 2, 3, 1)
        for i, (attn, ff) in enumerate(self.blocks):
            ctx = context_list[i] if context_list is not None else None
            x = attn(x, mask=mask, context_vec=ctx) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


# --- Остальные компоненты ---


class XiContextBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_gen = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.SiLU(),
                                         nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.local_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        res = x
        x = self.norm(x)
        context = self.context_gen(self.global_pool(x))
        return self.local_conv(x) * context + res


class DilatedKANLayer(nn.Module):

    def __init__(self, dim, dilation):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim, kernel_size=1)
        self.dw_conv = nn.Conv2d(dim,
                                 dim,
                                 3,
                                 padding=dilation,
                                 dilation=dilation,
                                 groups=dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.project_out(self.act(self.dw_conv(self.project_in(x)))) + x


class SK_Attention(nn.Module):

    def __init__(self, dim, branches=3):
        super().__init__()
        self.branches = branches
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(dim, dim // 4, 1), nn.SiLU(),
                                nn.Conv2d(dim // 4, dim * branches, 1))

    def forward(self, branch_list):
        combined = torch.stack(branch_list, dim=1)
        sum_feat = torch.sum(combined, dim=1)
        z = self.fc(self.pool(sum_feat))
        z = rearrange(z, 'b (br c) h w -> b br c h w', br=self.branches)
        weights = F.softmax(z, dim=1)
        return torch.sum(combined * weights, dim=1)


class ParallelChiNet(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.pre = nn.Conv2d(in_dim, hidden_dim, 1)
        self.branch1, self.branch2, self.branch4 = DilatedKANLayer(
            hidden_dim, 1), DilatedKANLayer(hidden_dim,
                                            2), DilatedKANLayer(hidden_dim, 4)
        self.sk_attn = SK_Attention(hidden_dim)
        self.post = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        x = self.pre(x)
        return self.post(
            self.sk_attn([self.branch1(x),
                          self.branch2(x),
                          self.branch4(x)]))


class GCE_v9(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_gaussians):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_vectors = num_gaussians + 1  # Эксперты + Арбитр
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * self.num_vectors))

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        v = self.pool(x).view(b, c)
        out = self.net(v).view(b, self.num_vectors, -1, 1, 1)
        return [out[:, i] for i in range(self.num_vectors)]


# --- Encoder v9 (Orchestra) ---


class EncoderV9(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_gaussians=7):
        super().__init__()
        self.num_gaussians = num_gaussians

        # hidden_dim обычно 32.
        # Для экспертов: 2 головы по 16 каналов (2 * 16 = 32)
        expert_heads_count = 2
        expert_dim_head = hidden_dim // expert_heads_count

        # Для арбитра: 1 голова на 32 канала (1 * 32 = 32)
        # Это делает арбитра "внимательнее" к общим связям
        orch_heads_count = 1
        orch_dim_head = hidden_dim // orch_heads_count

        self.gce = GCE_v9(in_dim, hidden_dim, num_gaussians)
        self.init_conv = nn.Conv2d(in_dim, hidden_dim, 3, padding=1)

        # Down-stage Backbone
        self.down = MSAB_Context(dim=hidden_dim,
                                 dim_head=hidden_dim // 2,
                                 heads=2,
                                 num_blocks=2,
                                 context_dim=0,
                                 mask_channels=in_dim)

        # 7 Экспертных голов
        self.expert_heads = nn.ModuleList([
            MSAB_Context(
                dim=hidden_dim,
                dim_head=expert_dim_head,
                heads=expert_heads_count,
                num_blocks=2,
                context_dim=hidden_dim,  # для GCE вектора
                mask_channels=in_dim) for _ in range(num_gaussians)
        ])

        self.expert_projs = nn.ModuleList([
            nn.Conv2d(hidden_dim, 3 * in_dim, 1) for _ in range(num_gaussians)
        ])

        # 1 Голова-Арбитр (Orchestrator)
        self.orchestrator = MSAB_Context(
            dim=hidden_dim,
            dim_head=orch_dim_head,
            heads=orch_heads_count,
            num_blocks=2,
            context_dim=hidden_dim,  # для GCE вектора
            mask_channels=in_dim)

        self.orch_proj = nn.Conv2d(hidden_dim, 2 * num_gaussians * in_dim, 1)

    def forward(self, x):
        ctx_list = self.gce(x)  # [Expert_1, ..., Expert_7, Orchestrator]
        feat = self.init_conv(x)
        feat = self.down(feat, x)

        # 1. Сбор данных от экспертов
        expert_params = []
        for i in range(self.num_gaussians):
            e_ctx = [ctx_list[i], ctx_list[i]]
            e_feat = self.expert_heads[i](feat, x, e_ctx)
            expert_params.append(self.expert_projs[i](e_feat))

        # 2. Сбор данных от арбитра
        o_ctx = [ctx_list[-1], ctx_list[-1]]
        orch_feat = self.orchestrator(feat, x, o_ctx)
        orch_params = self.orch_proj(orch_feat)

        return expert_params, orch_params


# --- HGSA v9 (Main Model) ---


class HGSA_v9(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, num_gaussians=7):
        super().__init__()
        self.num_gaussians, self.in_channels = num_gaussians, in_channels
        HIDDEN_FEAT = 32
        self.encoder = EncoderV9(in_channels, HIDDEN_FEAT, num_gaussians)
        self.xi_net = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1),
                                    XiContextBlock(16),
                                    nn.Conv2d(16, in_channels, 1),
                                    LayerNorm(in_channels))
        self.x_norm = LayerNorm(in_channels)
        self.chi_net = ParallelChiNet(in_channels * 3, HIDDEN_FEAT,
                                      out_channels)
        self.usgs_to_img = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        xi = self.xi_net(x)

        # Получаем параметры от Оркестра
        expert_data, orch_data = self.encoder(x)

        psi_total = torch.zeros_like(xi)
        p_list = []

        # --- ИСПРАВЛЕННЫЙ БЛОК НОРМАЛИЗАЦИИ ГЕЙТОВ ---
        # 1. Выделяем сырые гейты: [B, num_gaussians * C, H, W]
        raw_gates = orch_data[:, :self.num_gaussians * C]

        # 2. Решейпим для удобной нормализации: [B, num_gaussians, C, H, W]
        gates_reshaped = torch.sigmoid(raw_gates).view(B, self.num_gaussians,
                                                       C, H, W)

        # 3. Считаем сумму по оси экспертов (dim=1): [B, 1, C, H, W]
        gates_sum = gates_reshaped.sum(dim=1, keepdim=True) + 1e-6

        # 4. Нормализуем: каждый эксперт получает долю от общей энергии в этом канале
        gates_normed = gates_reshaped / gates_sum  # [B, num_gaussians, C, H, W]
        # ---------------------------------------------

        taus_all = torch.sigmoid(
            orch_data[:, self.num_gaussians * C:]) * 2.0 + 0.2

        for i in range(self.num_gaussians):
            # Параметры формы от эксперта i: [B, 3*C, H, W]
            e_p = expert_data[i]
            w = torch.tanh(e_p[:, :C]) * 1.5
            mu = torch.tanh(e_p[:, C:2 * C]) * 4.0
            sigma = torch.sigmoid(e_p[:, 2 * C:3 * C]) * 4.0 + 0.05

            # Параметры управления от арбитра (берем i-го эксперта из нормализованного тензора)
            gate = gates_normed[:, i]  # Достаем [B, C, H, W]
            tau = taus_all[:, i * C:(i + 1) * C]

            # USGS Аппроксимация
            diff = (xi - mu) / (sigma * tau + 1e-6)
            psi_i = gate * w * torch.exp(-0.5 * diff**2)
            psi_total = psi_total + psi_i

            if self.training:
                p_list.append(torch.stack([w, mu, sigma, gate, tau], dim=2))

        usgs_out = self.usgs_to_img(psi_total) + x
        combined = torch.cat([x, self.x_norm(x), psi_total], dim=1)
        main_out = self.chi_net(combined)

        return (main_out, usgs_out, p_list) if self.training else main_out
