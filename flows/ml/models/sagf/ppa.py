import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleContextExtractor(nn.Module):
    """Увеличенный Receptive Field через Dilated Convolutions"""
    def __init__(self, dim):
        super().__init__()
        self.atrous_block = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim) 
            for d in [1, 2, 4, 8]
        ])
        self.conv_1x1 = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x):
        res = [conv(x) for conv in self.atrous_block]
        return self.conv_1x1(torch.cat(res, dim=1))

class PentamodalAttention(nn.Module):
    def __init__(self, dim, num_concepts=5):
        super().__init__()
        self.num_concepts = num_concepts
        self.dim = dim
        
        # Обучаемые параметры V (не зависят от входа)
        # Представляют собой идеализированные цветовые трансформации
        self.v_parameters = nn.Parameter(torch.randn(1, num_concepts, dim, 1, 1))
        
        # Ветки генерации ключей (K)
        self.context_extractor = MultiScaleContextExtractor(dim)
        self.key_gen = nn.Conv2d(dim, dim * num_concepts, 1)
        
        # Усиленная Main Branch (Query)
        self.main_path = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(4, dim),
            nn.SiLU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Query: признаки основного изображения
        q = self.main_path(x) # [B, C, H, W]
        
        # 2. Keys: 5 веток модуляции из контекста
        context = self.context_extractor(x)
        keys = self.key_gen(context).view(b, self.num_concepts, c, h, w)
        keys = torch.softmax(keys, dim=1) # Распределение внимания между 5 состояниями
        
        # 3. Values: Обучаемые концепты, взвешенные ключами
        # Применяем Cross-Attention: V модулируется Keys и накладывается на Q
        modulated_v = torch.sum(keys * self.v_parameters, dim=1) # [B, C, H, W]
        
        # Финальная трансформация (Production-style: Residual + Multiplicative)
        out = q * (1 + modulated_v) 
        return out

class PPA_ColorMatcher(nn.Module):
    def __init__(self, in_channels=3, depth=6, width=64):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, width, 3, padding=1)
        
        # Многослойная структура с Residual связями
        self.layers = nn.ModuleList([
            PentamodalAttention(width) for _ in range(depth)
        ])
        
        self.final_refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.init_conv(x)
        for layer in self.layers:
            feat = feat + layer(feat) # Global Residual
        return self.final_refine(feat)