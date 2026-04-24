import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --- Базовые утилиты ---
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


class DWTForward(nn.Module):

    def forward(self, x):
        return F.pixel_unshuffle(x, 2)


# --- Компоненты Chi-Net (Multi-scale Dilated FFN) ---
class DilatedFFNLayer(nn.Module):

    def __init__(self, dim, dilation):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim, kernel_size=1)
        self.dw_conv = nn.Conv2d(dim,
                                 dim,
                                 kernel_size=3,
                                 padding=dilation,
                                 dilation=dilation,
                                 groups=dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        res = x
        x = self.project_in(x)
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.project_out(x)
        return x + res


# --- Компоненты Xi-Net ---
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
        self.branch1 = DilatedFFNLayer(hidden_dim, dilation=1)
        self.branch2 = DilatedFFNLayer(hidden_dim, dilation=2)
        self.branch4 = DilatedFFNLayer(hidden_dim, dilation=4)
        self.sk_attn = SK_Attention(hidden_dim)
        self.post = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        x = self.pre(x)
        b1, b2, b4 = self.branch1(x), self.branch2(x), self.branch4(x)
        return self.post(self.sk_attn([b1, b2, b4]))


# --- Компоненты Encoder2D (cmKAN style) ---
class IlluminationEstimator(nn.Module):

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1)
        self.depth_conv = nn.Conv2d(n_fea_middle,
                                    n_fea_middle,
                                    kernel_size=5,
                                    padding=2,
                                    groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        illu_fea = self.depth_conv(self.conv1(input))
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class Attention(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature_a = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_v = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_proj = nn.Conv2d(dim,
                                dim,
                                3,
                                padding=1,
                                stride=2,
                                groups=dim,
                                bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, 3, padding=1, stride=2, bias=bias)
        self.a_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, stride=2, groups=dim, bias=bias),
            nn.Conv2d(dim, dim // 2, kernel_size=1))
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illu_feat):
        b, c, h, w = x.shape
        q, k, a = self.q_proj(x), self.k_proj(x), self.a_proj(x)
        v = self.v_proj(x) * illu_feat
        q, k, a, v = [
            rearrange(t,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads) for t in (q, k, a, v)
        ]
        q, k, a = [F.normalize(t, dim=-1) for t in (q, k, a)]
        attn_a = (q @ a.transpose(-2, -1)) * self.temperature_a
        attn_k = (a @ k.transpose(-2, -1)) * self.temperature_v
        out = rearrange(attn_a.softmax(dim=-1) @ (attn_k.softmax(dim=-1) @ v),
                        'b head c (h w) -> b (head c) h w',
                        head=self.num_heads,
                        h=h,
                        w=w)
        return self.project_out(out)


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        self.out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features, hidden_features, 1)
        self.depthwise = nn.Conv2d(hidden_features,
                                   hidden_features,
                                   3,
                                   padding=1,
                                   groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features, self.out_features, 1)
        self.act_layer = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.pointwise2(
            self.act_layer(self.depthwise(self.pointwise1(x))))


class TransformerBlock(nn.Module):

    def __init__(self, in_channel, mid_channel, out_channel, num_heads, bias):
        super().__init__()
        self.norm1 = LayerNorm(in_channel)
        self.attn = Attention(in_channel, num_heads, bias)
        self.norm2 = LayerNorm(in_channel)
        self.ffn = FFN(in_channel, mid_channel, out_channel)

    def forward(self, x, illu_feat):
        x = x + self.attn(self.norm1(x), illu_feat)
        x = x + self.ffn(self.norm2(x))
        return x


class Encoder2D(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.estimator = IlluminationEstimator(12, in_dim + 1, in_dim)
        self.down1 = DWTForward()
        self.trans1 = TransformerBlock(12, 12, 12, 3, True)
        self.illu_down1 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(12, 12, 1))
        self.down2 = DWTForward()
        self.trans2 = TransformerBlock(48, 48, 48, 3, True)
        self.illu_down2 = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(12, 48, 1))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv_out = nn.Sequential(LayerNorm(in_dim + 12 + 48),
                                      FFN(in_dim + 12 + 48, out_dim, out_dim))

    def forward(self, x):
        illu_fea, illu_map = self.estimator(x)
        x_orig = x * illu_map + x
        x1 = self.trans1(self.down1(x_orig), self.illu_down1(illu_fea))
        x2 = self.trans2(self.down2(x1), self.illu_down2(illu_fea))
        out = torch.cat([x_orig, self.up1(x1), self.up2(x2)], dim=1)
        return self.conv_out(out)


# --- HGSA v10 Lite Orchestra ---
class HGSA_v10(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, num_gaussians=7):
        super().__init__()
        self.num_gaussians = num_gaussians
        HIDDEN_FEAT = 32
        XI_DIM = 16

        # Экстрактор признаков из v7
        self.encoder = Encoder2D(in_channels, out_dim=HIDDEN_FEAT)

        # Формирование аргумента xi
        self.xi_net = nn.Sequential(
            nn.Conv2d(in_channels, XI_DIM, 3, padding=1),
            XiContextBlock(XI_DIM), nn.Conv2d(XI_DIM, in_channels, 1),
            LayerNorm(in_channels))

        # --- v9 Orchestra Logic ---
        # 7 Экспертов: предсказывают только форму (w, mu, sigma) -> 3 параметра
        self.expert_heads = nn.ModuleList([
            FFN(HIDDEN_FEAT, HIDDEN_FEAT, 3 * in_channels)
            for _ in range(num_gaussians)
        ])

        # 1 Арбитр: предсказывает управление (gate, tau) для всех экспертов сразу
        # gate (1) + tau (1) = 2 параметра на эксперта на канал
        self.orchestrator = FFN(HIDDEN_FEAT, HIDDEN_FEAT * 2,
                                2 * num_gaussians * in_channels)

        self.x_norm = LayerNorm(in_channels)

        # Chi-Net (Multi-scale FFN)
        self.chi_net = ParallelChiNet(in_channels * 3, HIDDEN_FEAT,
                                      out_channels)

        # Голова для USGS Output
        self.usgs_to_img = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        xi = self.xi_net(x)
        feat = self.encoder(x)

        # 1. Получаем управляющие параметры от Арбитра
        orch_data = self.orchestrator(feat)

        # Разделяем на гейты и тау
        # Гейты прогоняем через Softmax для конкуренции экспертов
        raw_gates = orch_data[:, :self.num_gaussians * C].view(
            B, self.num_gaussians, C, H, W)
        gates_normed = F.softmax(raw_gates, dim=1)

        # Тау (растяжение)
        taus_all = torch.sigmoid(orch_data[:, self.num_gaussians * C:]).view(
            B, self.num_gaussians, C, H, W) * 2.0 + 0.2

        psi_total = torch.zeros_like(xi)
        p_list = []

        for i in range(self.num_gaussians):
            # 2. Получаем параметры формы от эксперта i
            p_e = self.expert_heads[i](feat).view(B, C, 3, H, W)

            w = torch.tanh(p_e[:, :, 0, :, :]) * 2.0
            mu = torch.tanh(p_e[:, :, 1, :, :]) * 4.0
            sigma = torch.sigmoid(p_e[:, :, 2, :, :]) * 4.0 + 0.05

            # 3. Применяем управление от Арбитра
            gate = gates_normed[:, i]
            tau = taus_all[:, i]

            # USGS Аппроксимация
            diff = (xi - mu) / (sigma * tau + 1e-6)
            psi_i = gate * w * torch.exp(-0.5 * diff**2)
            psi_total = psi_total + psi_i

            if self.training:
                # Сохраняем для TV-Loss (w, mu, sigma, gate, tau)
                p_list.append(torch.stack([w, mu, sigma, gate, tau], dim=2))

        usgs_out = self.usgs_to_img(psi_total) + x
        combined = torch.cat([x, self.x_norm(x), psi_total], dim=1)
        main_out = self.chi_net(combined)

        return (main_out, usgs_out, p_list) if self.training else main_out
