"""
CVAE-GAN Hybrid Fusion Generator
==================================
Combines VAE structured latent space with GAN adversarial sharpness.
Triple loss: Reconstruction + KL divergence + Adversarial.

Reference: Larsen et al., "Autoencoding beyond pixels using a learned
           similarity metric", 2016
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import ResidualBlock, SelfAttention, init_weights


class FusionEncoder(nn.Module):
    """VAE-style encoder producing (mu, logvar)."""

    def __init__(self, in_channels: int = 1, channels: list[int] | None = None,
                 latent_dim: int = 128, image_size: int = 32) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out in channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch_out), nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(ch_out, ch_out),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        self._flat = channels[-1] * (image_size // (2 ** len(channels))) ** 2
        self.fc_mu = nn.Linear(self._flat, latent_dim)
        self.fc_logvar = nn.Linear(self._flat, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class FusionDecoder(nn.Module):
    """GAN-style generator from latent to image."""

    def __init__(self, out_channels: int = 1, channels: list[int] | None = None,
                 latent_dim: int = 128, image_size: int = 32) -> None:
        super().__init__()
        channels = channels or [256, 128, 64, 32]
        n = len(channels)
        self._is = image_size // (2 ** n)
        self._ic = channels[0]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, channels[0] * self._is ** 2),
            nn.BatchNorm1d(channels[0] * self._is ** 2), nn.ReLU(True),
        )
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i+1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels[i+1]), nn.ReLU(True),
            ])
            if i == 1:
                layers.append(SelfAttention(channels[i+1]))
        layers.extend([
            nn.ConvTranspose2d(channels[-1], out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        ])
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        h = self.fc(z).view(z.size(0), self._ic, self._is, self._is)
        return self.deconv(h)


class FusionDiscriminator(nn.Module):
    """Discriminator for the fusion model."""

    def __init__(self, in_channels: int = 1, channels: list[int] | None = None,
                 image_size: int = 32) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        layers: list[nn.Module] = []
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            layers.extend([
                nn.utils.spectral_norm(
                    nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False)),
                nn.GroupNorm(1, ch_out), nn.LeakyReLU(0.2, True),
            ])
            if i == 1:
                layers.append(SelfAttention(ch_out))
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        fs = image_size // (2 ** len(channels))
        self._flat = channels[-1] * fs ** 2
        self.out = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self._flat, 256)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Linear(256, 1)), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        h = self.conv(x).view(x.size(0), -1)
        return self.out(h)


class CVAEGANFusion(nn.Module):
    """Combined CVAE-GAN: VAE encoder + GAN decoder + discriminator.

    Triple loss = recon_weight * Recon + kl_weight * KL + adv_weight * Adv
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 128,
                 encoder_channels: list[int] | None = None,
                 decoder_channels: list[int] | None = None,
                 d_channels: list[int] | None = None,
                 image_size: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = FusionEncoder(in_channels, encoder_channels, latent_dim, image_size)
        self.decoder = FusionDecoder(in_channels, decoder_channels, latent_dim, image_size)
        self.discriminator = FusionDiscriminator(in_channels, d_channels, image_size)
        init_weights(self, strategy="he")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def generate(self, num_samples: int, device: torch.device):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)

    @staticmethod
    def compute_losses(recon, target, mu, logvar, d_real, d_fake,
                       recon_w=10.0, kl_w=1.0, adv_w=1.0):
        target_01 = (target + 1.0) / 2.0
        recon_loss = F.binary_cross_entropy(
            (recon + 1.0) / 2.0, target_01, reduction="sum") / target.size(0)
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
        d_loss = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
        g_loss = -torch.mean(torch.log(d_fake + 1e-8))
        enc_dec_loss = recon_w * recon_loss + kl_w * kl_loss + adv_w * g_loss
        return {"enc_dec_loss": enc_dec_loss, "d_loss": d_loss,
                "recon_loss": recon_loss, "kl_loss": kl_loss,
                "g_loss": g_loss}
