"""
Conditional GAN Trainer
========================
Class-conditioned training with per-class quality tracking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.conditional_gan import ConditionalGenerator, ProjectionDiscriminator
from src.training.base_trainer import BaseTrainer
from src.utils.config import Config


class CGANTrainer(BaseTrainer):

    def __init__(self, config: Config, device: torch.device | None = None):
        super().__init__(config, "cgan", device)

    def setup(self, **kwargs) -> None:
        ccfg = self.config.models.cgan
        latent_dim = self.config.models.latent_dim
        num_classes = self.config.data.num_classes

        self.generator = ConditionalGenerator(
            latent_dim=latent_dim, num_classes=num_classes,
            embed_dim=getattr(ccfg, "embed_dim", 64),
            channels=ccfg.g_channels,
            out_channels=self.config.data.channels,
            image_size=self.config.data.image_size,
        ).to(self.device)

        self.discriminator = ProjectionDiscriminator(
            in_channels=self.config.data.channels,
            num_classes=num_classes,
            channels=ccfg.d_channels,
            image_size=self.config.data.image_size,
        ).to(self.device)

        opt_cfg = self.config.training.optimizer
        lr = getattr(opt_cfg, "lr", 2e-4)
        betas = tuple(getattr(opt_cfg, "betas", [0.5, 0.999]))

        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        self.criterion = nn.BCELoss()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Fixed noise + labels for consistent visualization
        self.fixed_z = torch.randn(num_classes * 8, latent_dim, device=self.device)
        self.fixed_labels = torch.arange(num_classes, device=self.device).repeat(8)

        g_p = sum(p.numel() for p in self.generator.parameters())
        d_p = sum(p.numel() for p in self.discriminator.parameters())
        self.logger.info(f"cGAN Generator: {g_p:,} | Discriminator: {d_p:,}")

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        g_total = d_total = 0.0
        n = 0

        for images, labels in tqdm(dataloader, desc="Train cGAN", leave=False):
            bs = images.size(0)
            images, labels = images.to(self.device), labels.to(self.device)

            real_target = torch.ones(bs, 1, device=self.device) * 0.9
            fake_target = torch.ones(bs, 1, device=self.device) * 0.1

            # ── Discriminator ──
            self.opt_d.zero_grad()
            with self.autocast():
                d_real = self.discriminator(images, labels)
            with self.autocast(enabled=False):
                loss_d_real = self.criterion(d_real.float(), real_target.float())

            z = torch.randn(bs, self.latent_dim, device=self.device)
            with self.autocast():
                fake = self.generator(z, labels).detach()
                d_fake = self.discriminator(fake, labels)
            with self.autocast(enabled=False):
                loss_d_fake = self.criterion(d_fake.float(), fake_target.float())

            loss_d = (loss_d_real + loss_d_fake) / 2
            self.backward_step(
                loss_d,
                self.opt_d,
                clip_params=self.discriminator.parameters(),
                clip_value=self.config.training.gradient_clip,
                scaler_update=False,
            )

            # ── Generator ──
            self.opt_g.zero_grad()
            z = torch.randn(bs, self.latent_dim, device=self.device)
            with self.autocast():
                fake = self.generator(z, labels)
                d_fake_g = self.discriminator(fake, labels)
            with self.autocast(enabled=False):
                loss_g = self.criterion(d_fake_g.float(), torch.ones(bs, 1, device=self.device).float())
            self.backward_step(
                loss_g,
                self.opt_g,
                clip_params=self.generator.parameters(),
                clip_value=self.config.training.gradient_clip,
                scaler_update=True,
            )

            g_total += loss_g.item()
            d_total += loss_d.item()
            n += 1
            self.global_step += 1

        return {"g_loss": g_total / n, "d_loss": d_total / n}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.discriminator.eval()
        d_total = 0.0
        n = 0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            with self.autocast(enabled=False):
                d_real = self.discriminator(images, labels)
                d_total += self.criterion(d_real.float(), torch.ones_like(d_real).float()).item()
            n += 1
        return {"d_loss": d_total / n}

    @torch.no_grad()
    def generate_samples(self, epoch: int) -> None:
        self.generator.eval()
        samples = self.generator(self.fixed_z, self.fixed_labels)
        grid = vutils.make_grid(samples, nrow=self.num_classes, normalize=True, padding=2)
        self.logger.log_images("cgan/generated", samples, epoch)
        vutils.save_image(grid, self.generated_dir / f"cgan_epoch_{epoch:04d}.png")

    def _save_checkpoint(self, epoch, metrics):
        models = {"generator": self.generator, "discriminator": self.discriminator}
        optims = {"opt_g": self.opt_g, "opt_d": self.opt_d}
        self.ckpt_mgr.save(models, optims, epoch, metrics)
        self.ckpt_mgr.save_best(models, optims, epoch,
                                metrics.get("g_loss", float("inf")), metrics)
