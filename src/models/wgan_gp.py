"""
Wasserstein GAN with Gradient Penalty (WGAN-GP)
=================================================
Critic (no sigmoid) trained with gradient penalty instead of weight
clipping, producing more stable training and meaningful loss curves.

Key differences from DCGAN:
- Critic outputs unbounded scalar (no Sigmoid)
- No BatchNorm in Critic — uses LayerNorm (GroupNorm(1))
- Gradient penalty (λ=10) enforces 1-Lipschitz constraint
- Critic trained n_critic times per generator step

Reference: Gulrajani et al., "Improved Training of Wasserstein GANs", 2017
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.autograd as autograd

from src.models.components import SelfAttention, MinibatchStdDev, init_weights


class WGANGenerator(nn.Module):
    """Wasserstein GAN Generator (same architecture as DCGAN Generator)."""

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
        self._init_size = image_size // (2 ** n_up)
        self._init_channels = channels[0]

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
            if i == 1:
                layers.append(SelfAttention(channels[i + 1]))

        layers.extend([
            nn.ConvTranspose2d(
                channels[-1], out_channels, 4, stride=2, padding=1, bias=False,
            ),
            nn.Tanh(),
        ])

        self.deconv_stack = nn.Sequential(*layers)
        init_weights(self, strategy="orthogonal")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.project(z)
        h = h.view(h.size(0), self._init_channels, self._init_size, self._init_size)
        return self.deconv_stack(h)


class WGANCritic(nn.Module):
    """Wasserstein Critic — NO Sigmoid, NO BatchNorm.

    Uses LayerNorm (via GroupNorm(1)) and LeakyReLU for stable
    Wasserstein distance estimation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | None = None,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]

        layers: list[nn.Module] = []
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            layers.extend([
                nn.Conv2d(ch_in, ch_out, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, ch_out),   # LayerNorm equivalent
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if i == 1:
                layers.append(SelfAttention(ch_out))
            ch_in = ch_out

        layers.append(MinibatchStdDev())
        self.conv_stack = nn.Sequential(*layers)

        final_size = image_size // (2 ** len(channels))
        self._flat_size = (channels[-1] + 1) * final_size ** 2

        # No Sigmoid — outputs unbounded Wasserstein distance estimate
        self.output = nn.Sequential(
            nn.Linear(self._flat_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
        init_weights(self, strategy="orthogonal")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_stack(x)
        h = h.view(h.size(0), -1)
        return self.output(h)


def compute_gradient_penalty(
    critic: WGANCritic,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Compute the gradient penalty for WGAN-GP.

    Enforces the 1-Lipschitz constraint by penalising gradients of the
    critic's output with respect to interpolated inputs.

    Parameters
    ----------
    critic : WGANCritic
        The critic network.
    real_samples, fake_samples : Tensor
        Batches of real and generated images.
    device : torch.device
        Compute device.
    lambda_gp : float
        Gradient penalty coefficient (default 10).

    Returns
    -------
    gradient_penalty : Tensor (scalar)
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    critic_interpolated = critic(interpolated)

    gradients = autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1.0) ** 2).mean()
    return penalty
