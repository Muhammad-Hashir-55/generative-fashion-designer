"""
Deep Convolutional GAN (DCGAN)
===============================
Generator and Discriminator with spectral normalization, self-attention,
and minibatch standard deviation for Fashion-MNIST.

Architecture (32×32 grayscale)
------------------------------
- Generator:  z(128) → FC → 2×2×256 → [↑ ConvT → BN → ReLU] ×4 → Tanh
- Discriminator: [Conv → LN → LeakyReLU] ×4 → MinibatchStd → FC → Sigmoid

Reference: Radford et al., "Unsupervised Representation Learning with
           Deep Convolutional Generative Adversarial Networks", 2016
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

from src.models.components import (
    SelfAttention,
    MinibatchStdDev,
    init_weights,
)


class DCGenerator(nn.Module):
    """DCGAN Generator: latent vector → 32×32 image.

    Parameters
    ----------
    latent_dim : int
        Size of the input noise vector *z*.
    channels : list[int]
        Channel counts for each upsampling stage.
    out_channels : int
        Output image channels (1 for grayscale).
    image_size : int
        Target spatial resolution.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        channels: list[int] | None = None,
        out_channels: int = 1,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        channels = channels or [256, 128, 64, 32]

        n_up = len(channels)
        self._init_size = image_size // (2 ** n_up)    # 32 / 16 = 2
        self._init_channels = channels[0]

        # Project noise to initial feature map
        self.project = nn.Sequential(
            nn.Linear(latent_dim, channels[0] * self._init_size ** 2),
            nn.BatchNorm1d(channels[0] * self._init_size ** 2),
            nn.ReLU(inplace=True),
        )

        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    channels[i], channels[i + 1], 4, stride=2, padding=1, bias=False,
                ),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ])
            # Insert self-attention after second upsampling (8×8 resolution)
            if i == 1:
                layers.append(SelfAttention(channels[i + 1]))

        # Final upsampling to full resolution
        layers.extend([
            nn.ConvTranspose2d(
                channels[-1], out_channels, 4, stride=2, padding=1, bias=False,
            ),
            nn.Tanh(),
        ])

        self.deconv_stack = nn.Sequential(*layers)
        init_weights(self, strategy="he")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.project(z)
        h = h.view(h.size(0), self._init_channels, self._init_size, self._init_size)
        return self.deconv_stack(h)


class DCDiscriminator(nn.Module):
    """DCGAN Discriminator with spectral normalisation.

    Parameters
    ----------
    in_channels : int
        Input image channels.
    channels : list[int]
        Channel counts for each downsampling stage.
    image_size : int
        Input spatial resolution.
    use_spectral_norm : bool
        Apply spectral normalisation to all conv/linear layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | None = None,
        image_size: int = 32,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]

        def maybe_sn(layer: nn.Module) -> nn.Module:
            return spectral_norm(layer) if use_spectral_norm else layer

        layers: list[nn.Module] = []
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            layers.append(
                maybe_sn(nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1, bias=False))
            )
            # Use LayerNorm instead of BatchNorm for discriminator stability
            layers.append(nn.GroupNorm(1, ch_out))   # equivalent to LayerNorm on channels
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.25))
            # Self-attention at 8×8 resolution
            if i == 1:
                layers.append(SelfAttention(ch_out))
            ch_in = ch_out

        layers.append(MinibatchStdDev())

        self.conv_stack = nn.Sequential(*layers)

        # +1 for MinibatchStdDev channel
        final_size = image_size // (2 ** len(channels))
        self._flat_size = (channels[-1] + 1) * final_size ** 2

        self.classifier = nn.Sequential(
            maybe_sn(nn.Linear(self._flat_size, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            maybe_sn(nn.Linear(256, 1)),
            nn.Sigmoid(),
        )
        init_weights(self, strategy="he")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_stack(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)
