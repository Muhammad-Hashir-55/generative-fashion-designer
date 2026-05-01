"""
Shared Neural Network Building Blocks
=======================================
Reusable components used across all generative model architectures:
- Residual blocks with optional squeeze-and-excitation
- Self-attention layers for feature maps
- Spectral normalization wrapper
- Learnable class embeddings for conditional models
- Weight initialization utilities
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  Residual Block
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Conv → Norm → ReLU → Conv → Norm + skip connection.

    If ``in_channels != out_channels``, a 1×1 convolution is used for the
    skip projection so that dimensions align.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "batch",
        activation: str = "relu",
    ) -> None:
        super().__init__()

        NormLayer = nn.BatchNorm2d if norm == "batch" else nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            NormLayer(out_channels),
            _activation(activation),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            NormLayer(out_channels),
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.skip(x))


# ═══════════════════════════════════════════════════════════════════════════════
#  Squeeze-and-Excitation Block
# ═══════════════════════════════════════════════════════════════════════════════

class SqueezeExcitation(nn.Module):
    """Channel attention via global average pooling → FC → Sigmoid gating."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-Attention Layer (SA-GAN style)
# ═══════════════════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    """Self-attention mechanism for 2-D feature maps (Zhang et al., 2019).

    Computes query-key-value attention over spatial positions to capture
    long-range dependencies that convolutions miss.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)   # B × N × C'
        k = self.key(x).view(B, -1, H * W)                       # B × C' × N
        v = self.value(x).view(B, -1, H * W)                     # B × C  × N

        attn = torch.bmm(q, k)                                   # B × N × N
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))                # B × C × N
        out = out.view(B, C, H, W)

        return self.gamma * out + x


# ═══════════════════════════════════════════════════════════════════════════════
#  Class Embedding
# ═══════════════════════════════════════════════════════════════════════════════

class ClassEmbedding(nn.Module):
    """Learnable embedding table for class-conditional generation."""

    def __init__(self, num_classes: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embed(labels)


# ═══════════════════════════════════════════════════════════════════════════════
#  Weight Initialization
# ═══════════════════════════════════════════════════════════════════════════════

def init_weights(module: nn.Module, strategy: str = "he") -> None:
    """Apply weight initialization to Conv2d and Linear layers.

    Parameters
    ----------
    module : nn.Module
        The module tree to initialise (applied recursively).
    strategy : str
        ``"he"`` (Kaiming), ``"xavier"``, or ``"orthogonal"``.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if strategy == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif strategy == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif strategy == "orthogonal":
                nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ═══════════════════════════════════════════════════════════════════════════════
#  Minibatch Standard Deviation (for discriminator)
# ═══════════════════════════════════════════════════════════════════════════════

class MinibatchStdDev(nn.Module):
    """Appends the minibatch standard deviation as an extra feature map.

    Encourages the discriminator to use batch-level statistics to
    detect mode collapse (Karras et al., 2018).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=0, keepdim=True).mean()
        std_map = std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std_map], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    activations = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        "elu": nn.ELU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations)}")
    return activations[name]
