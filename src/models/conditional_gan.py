"""
Conditional GAN (cGAN) with Projection Discriminator
======================================================
Class-conditional generation enabling targeted synthesis of specific
clothing categories (e.g., "generate a Dress").

Architecture
------------
- Generator: Class embedding concatenated to noise z → ConvT stack
- Discriminator: Projection discriminator (Miyato & Koyama, 2018) —
  inner product between class embedding and feature vector replaces
  naive concatenation for better gradient flow.

Reference: Mirza & Osindero, "Conditional GANs", 2014
           Miyato & Koyama, "cGANs with Projection Discriminator", 2018
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

from src.models.components import SelfAttention, init_weights


class ConditionalGenerator(nn.Module):
    """Class-conditional generator: (z, class_label) → image.

    The class label is mapped to an embedding vector which is
    concatenated with the noise vector before generation.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 10,
        embed_dim: int = 64,
        channels: list[int] | None = None,
        out_channels: int = 1,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        channels = channels or [256, 128, 64, 32]
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_embed = nn.Embedding(num_classes, embed_dim)

        n_up = len(channels)
        self._init_size = image_size // (2 ** n_up)
        self._init_channels = channels[0]

        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, channels[0] * self._init_size ** 2),
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
            if i == 1:
                layers.append(SelfAttention(channels[i + 1]))

        layers.extend([
            nn.ConvTranspose2d(
                channels[-1], out_channels, 4, stride=2, padding=1, bias=False,
            ),
            nn.Tanh(),
        ])

        self.deconv_stack = nn.Sequential(*layers)
        init_weights(self, strategy="he")

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate images conditioned on class labels.

        Parameters
        ----------
        z : Tensor [B, latent_dim]
            Random noise vector.
        labels : Tensor [B]
            Integer class labels ∈ {0, ..., num_classes - 1}.
        """
        embed = self.label_embed(labels)             # [B, embed_dim]
        z_cond = torch.cat([z, embed], dim=1)        # [B, latent_dim + embed_dim]
        h = self.project(z_cond)
        h = h.view(h.size(0), self._init_channels, self._init_size, self._init_size)
        return self.deconv_stack(h)


class ProjectionDiscriminator(nn.Module):
    """Projection Discriminator (Miyato & Koyama, 2018).

    Instead of concatenating class info, computes:
        D(x, y) = σ(φ(x)ᵀ · e(y) + ψ(φ(x)))

    where φ(x) are image features, e(y) is the class embedding,
    and ψ is a linear classifier.  This produces better gradients
    than concatenation-based conditioning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: list[int] | None = None,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]

        layers: list[nn.Module] = []
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            layers.extend([
                spectral_norm(
                    nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1, bias=False)
                ),
                nn.GroupNorm(1, ch_out),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
            ])
            if i == 1:
                layers.append(SelfAttention(ch_out))
            ch_in = ch_out

        self.conv_stack = nn.Sequential(*layers)

        final_size = image_size // (2 ** len(channels))
        self._flat_size = channels[-1] * final_size ** 2
        feature_dim = 256

        # Feature extraction
        self.feature_fc = nn.Sequential(
            spectral_norm(nn.Linear(self._flat_size, feature_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Projection: class embedding for inner-product conditioning
        self.class_embed = spectral_norm(
            nn.Embedding(num_classes, feature_dim)
        )

        # Unconditional logit
        self.output_fc = spectral_norm(nn.Linear(feature_dim, 1))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Classify real/fake conditioned on class labels.

        Parameters
        ----------
        x : Tensor [B, C, H, W]
            Input images.
        labels : Tensor [B]
            Class labels.

        Returns
        -------
        logits : Tensor [B, 1]
            Unbounded logits (apply Sigmoid externally for BCE).
        """
        h = self.conv_stack(x)
        h = h.view(h.size(0), -1)
        features = self.feature_fc(h)                    # [B, feature_dim]

        # Projection conditioning
        class_emb = self.class_embed(labels)             # [B, feature_dim]
        projection = (features * class_emb).sum(dim=1, keepdim=True)  # [B, 1]

        unconditional = self.output_fc(features)         # [B, 1]

        return projection + unconditional  # raw logits, no sigmoid (AMP-safe)
