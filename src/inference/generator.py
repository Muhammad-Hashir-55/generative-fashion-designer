"""
Unified Fashion Generator
==========================
Single interface for loading any trained model and generating samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.vae import VAE
from src.models.dcgan import DCGenerator
from src.models.wgan_gp import WGANGenerator
from src.models.conditional_gan import ConditionalGenerator
from src.models.fusion_generator import CVAEGANFusion
from src.utils.config import Config
from src.utils.checkpoint import CheckpointManager
from src.data.dataset import FASHION_CLASSES, name_to_label


class FashionGenerator:
    """Load any trained model and generate fashion images.

    Usage
    -----
    >>> gen = FashionGenerator(config, model_type="vae")
    >>> gen.load_checkpoint("outputs/checkpoints/vae_best.pt")
    >>> images = gen.generate(16)
    """

    MODEL_REGISTRY = {
        "vae": "vae",
        "dcgan": "generator",
        "wgan_gp": "generator",
        "cgan": "generator",
        "fusion": "decoder",
    }

    def __init__(self, config: Config, model_type: str = "vae",
                 device: torch.device | None = None) -> None:
        self.config = config
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = config.models.latent_dim
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> torch.nn.Module:
        cfg = self.config
        if self.model_type == "vae":
            return VAE(
                in_channels=cfg.data.channels, latent_dim=cfg.models.latent_dim,
                encoder_channels=cfg.models.vae.encoder_channels,
                decoder_channels=cfg.models.vae.decoder_channels,
                image_size=cfg.data.image_size)
        elif self.model_type == "dcgan":
            return DCGenerator(
                latent_dim=cfg.models.latent_dim,
                channels=cfg.models.dcgan.g_channels,
                out_channels=cfg.data.channels,
                image_size=cfg.data.image_size)
        elif self.model_type == "wgan_gp":
            return WGANGenerator(
                latent_dim=cfg.models.latent_dim,
                channels=cfg.models.wgan_gp.g_channels,
                out_channels=cfg.data.channels,
                image_size=cfg.data.image_size)
        elif self.model_type == "cgan":
            return ConditionalGenerator(
                latent_dim=cfg.models.latent_dim,
                num_classes=cfg.data.num_classes,
                embed_dim=getattr(cfg.models.cgan, "embed_dim", 64),
                channels=cfg.models.cgan.g_channels,
                out_channels=cfg.data.channels,
                image_size=cfg.data.image_size)
        elif self.model_type == "fusion":
            return CVAEGANFusion(
                in_channels=cfg.data.channels, latent_dim=cfg.models.latent_dim,
                image_size=cfg.data.image_size)
        raise ValueError(f"Unknown model type: {self.model_type}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load weights from a checkpoint file."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state" in state:
            self.model.load_state_dict(state["model_state"])
        elif "model_states" in state:
            key = self.MODEL_REGISTRY.get(self.model_type, "generator")
            if key in state["model_states"]:
                self.model.load_state_dict(state["model_states"][key])
        print(f"Loaded checkpoint from {path}")

    @torch.no_grad()
    def generate(self, num_samples: int = 16,
                 class_label: str | int | None = None) -> torch.Tensor:
        """Generate fashion images.

        Parameters
        ----------
        num_samples : int
        class_label : str or int, optional
            For conditional GAN: specify a class (name or index).
        """
        self.model.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)

        if self.model_type == "vae":
            return self.model.decode(z)
        elif self.model_type == "cgan":
            if class_label is None:
                labels = torch.randint(0, self.config.data.num_classes,
                                       (num_samples,), device=self.device)
            else:
                if isinstance(class_label, str):
                    class_label = name_to_label(class_label)
                labels = torch.full((num_samples,), class_label,
                                    dtype=torch.long, device=self.device)
            return self.model(z, labels)
        elif self.model_type == "fusion":
            return self.model.generate(num_samples, self.device)
        else:
            return self.model(z)

    @torch.no_grad()
    def interpolate(self, z1: torch.Tensor | None = None,
                    z2: torch.Tensor | None = None,
                    steps: int = 10) -> torch.Tensor:
        """Generate interpolation between two latent vectors."""
        self.model.eval()
        if z1 is None:
            z1 = torch.randn(1, self.latent_dim, device=self.device)
        if z2 is None:
            z2 = torch.randn(1, self.latent_dim, device=self.device)

        alphas = torch.linspace(0, 1, steps, device=self.device)
        images = []
        for a in alphas:
            z = (1 - a) * z1 + a * z2
            if self.model_type == "vae":
                img = self.model.decode(z)
            elif self.model_type == "fusion":
                img = self.model.decoder(z)
            else:
                img = self.model(z)
            images.append(img)
        return torch.cat(images, dim=0)
