import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedVibLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Learnable physics parameters per channel
        self.omega = nn.Parameter(torch.ones(1, channels, 1, 1) * 1.5)
        self.zeta = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        
        # Spatial coupling (Depthwise)
        self.coupling = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        # Non-linear activation for the 'spring' constant
        self.softplus = nn.Softplus()

    def forward(self, force, steps=6, dt=0.2):
        x = torch.zeros_like(force)
        v = torch.zeros_like(force)
        
        # Ensure physics parameters stay positive
        w = self.softplus(self.omega)
        z = torch.sigmoid(self.zeta) # Damping between 0 and 1
        
        for _ in range(steps):
            interaction = self.coupling(x)
            # Non-linear oscillator: force is modulated by current displacement
            dv = (-2 * z * w * v - (w**2) * x + force + interaction) * dt
            v = v + dv
            x = x + v * dt
            
        return 0.5 * (v**2) + 0.5 * (w**2) * (x**2)

class DeepVibrativeMatcherV2(nn.Module):
    def __init__(self, channels=16, num_layers=4):
        super().__init__()
        self.input_proj = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        
        # 1. Hierarchical Vib Stack with Skip Connections
        self.vib_layers = nn.ModuleList([
            AdvancedVibLayer(channels) for _ in range(num_layers)
        ])
        
        # 2. Cross-Channel Vibrant Attention
        # Learns which color channels should 'vibrate' together
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)
        
        # Initial projection
        feat = self.input_proj(x)
        
        # Resonant Backbone with Residual Energy
        for vib_layer in self.vib_layers:
            # Residual connection: adds driving force to the next layer's energy
            feat = feat + vib_layer(feat)
            
        # Channel-wise attention (Global Color Context)
        feat = feat * self.channel_attn(feat)
        
        return self.tail(feat)