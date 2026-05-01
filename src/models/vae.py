"""
Variational Autoencoder (VAE)
==============================
Convolutional β-VAE with residual encoder, KL annealing support,
and transposed-convolution decoder for Fashion-MNIST generation.

Architecture
------------
- Encoder: 4 conv layers (stride 2) with residual connections → (μ, log σ²)
- Reparameterisation trick: z = μ + σ ⊙ ε,  ε ~ N(0, I)
- Decoder: FC → reshape → 4 transposed conv layers → Sigmoid
- Loss: BCE reconstruction + β · KL(q(z|x) ‖ p(z))

Reference: Kingma & Welling, "Auto-Encoding Variational Bayes", 2014
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from src.models.components import ResidualBlock, init_weights


class Encoder(nn.Module):
    """Convolutional encoder mapping images → (μ, log σ²)."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | None = None,
        latent_dim: int = 128,
        image_size: int = 32,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]

        layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out in channels:
            layers.append(
                nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if use_residual:
                layers.append(ResidualBlock(ch_out, ch_out))
            ch_in = ch_out

        self.conv_stack = nn.Sequential(*layers)

        # Compute flattened size after conv stack
        self._flat_size = channels[-1] * (image_size // (2 ** len(channels))) ** 2

        self.fc_mu     = nn.Linear(self._flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_stack(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Transposed-convolution decoder mapping z → reconstructed image."""

    def __init__(
        self,
        out_channels: int = 1,
        channels: list[int] | None = None,
        latent_dim: int = 128,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        channels = channels or [256, 128, 64, 32]

        n_up = len(channels)
        self._init_size = image_size // (2 ** n_up)
        self._init_channels = channels[0]

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, channels[0] * self._init_size ** 2),
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

        # Final layer to image space
        layers.extend([
            nn.ConvTranspose2d(
                channels[-1], out_channels, 4, stride=2, padding=1, bias=False,
            ),
            nn.Sigmoid(),
        ])

        self.deconv_stack = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), self._init_channels, self._init_size, self._init_size)
        return self.deconv_stack(h)


class VAE(nn.Module):
    """Full Variational Autoencoder.

    Parameters
    ----------
    in_channels : int
        Image channels (1 for grayscale Fashion-MNIST).
    latent_dim : int
        Dimensionality of the latent vector *z*.
    encoder_channels : list[int]
        Channel progression for the encoder conv stack.
    decoder_channels : list[int]
        Channel progression for the decoder deconv stack.
    image_size : int
        Spatial size of input images (assumes square).
    use_residual : bool
        Whether to add residual blocks inside the encoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        encoder_channels: list[int] | None = None,
        decoder_channels: list[int] | None = None,
        image_size: int = 32,
        use_residual: bool = True,
        use_pretrained_encoder: bool = False,
        freeze_pretrained_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.use_pretrained_encoder = use_pretrained_encoder

        if use_pretrained_encoder:
            weights = ResNet18_Weights.IMAGENET1K_V1
            backbone = resnet18(weights=weights)
            # Keep convolutional trunk only; output is [B, 512, 1, 1] with adaptive pool.
            self.pretrained_encoder = nn.Sequential(*list(backbone.children())[:-1])
            if in_channels != 3:
                raise ValueError("Pretrained ResNet encoder requires 3-channel RGB input")
            if freeze_pretrained_encoder:
                for p in self.pretrained_encoder.parameters():
                    p.requires_grad = False
            self.fc_mu = nn.Linear(512, latent_dim)
            self.fc_logvar = nn.Linear(512, latent_dim)
            self.encoder = None
        else:
            self.encoder = Encoder(
                in_channels, encoder_channels, latent_dim, image_size, use_residual,
            )
            self.pretrained_encoder = None
            self.fc_mu = None
            self.fc_logvar = None

        self.decoder = Decoder(
            in_channels, decoder_channels, latent_dim, image_size,
        )

        init_weights(self, strategy="he")

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample z via the reparameterisation trick: z = μ + σ ⊙ ε."""
        # Clamp logvar for numerical stability to prevent NaNs
        logvar = torch.clamp(logvar, -10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        if self.use_pretrained_encoder:
            h = self.pretrained_encoder(x)
            h = h.view(h.size(0), -1)
            return self.fc_mu(h), self.fc_logvar(h)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterize → decode.

        Returns ``(reconstruction, mu, logvar)``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new samples by decoding random latent vectors."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @staticmethod
    def loss_function(
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute β-VAE loss = Reconstruction + β · KL divergence.

        Parameters
        ----------
        recon : Tensor
            Reconstructed images from the decoder.
        target : Tensor
            Original input images (scaled to [0, 1] for BCE).
        mu, logvar : Tensor
            Encoder outputs.
        beta : float
            KL weight (β = 1 is standard VAE; β > 1 is β-VAE).

        Returns
        -------
        total_loss, recon_loss, kl_loss
        """
        # Support two common input normalizations:
        # - GAN-style: target in [-1, 1] -> scale to [0, 1]
        # - ImageNet-style: target normalized by (x - mean)/std -> invert
        if target.min() >= -1.01 and target.max() <= 1.01:
            target_01 = (target + 1.0) / 2.0
        else:
            mean = torch.tensor([0.485, 0.456, 0.406], device=target.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=target.device).view(1, 3, 1, 1)
            target_01 = target * std + mean
            target_01 = torch.clamp(target_01, 0.0, 1.0)

        # Use MSE for RGB reconstruction stability (works with any scaling)
        recon_loss = F.mse_loss(recon, target_01, reduction="sum") / target.size(0)

        # KL divergence: -0.5 * Σ(1 + log σ² - μ² - σ²)
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / target.size(0)

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss
