"""
DCGAN Trainer
==============
Two-optimizer training loop for DCGAN with label smoothing,
noise injection, discriminator accuracy tracking, and fixed-noise
sample generation for visual progress monitoring.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.dcgan import DCGenerator, DCDiscriminator
from src.training.base_trainer import BaseTrainer
from src.training.scheduler import build_scheduler
from src.utils.config import Config


class GANTrainer(BaseTrainer):

    def __init__(self, config: Config, device: torch.device | None = None):
        super().__init__(config, "dcgan", device)

    def setup(self, **kwargs) -> None:
        dcfg = self.config.models.dcgan
        latent_dim = self.config.models.latent_dim

        self.generator = DCGenerator(
            latent_dim=latent_dim,
            channels=dcfg.g_channels,
            out_channels=self.config.data.channels,
            image_size=self.config.data.image_size,
        ).to(self.device)

        self.discriminator = DCDiscriminator(
            in_channels=self.config.data.channels,
            channels=dcfg.d_channels,
            image_size=self.config.data.image_size,
        ).to(self.device)

        opt_cfg = self.config.training.optimizer
        lr = getattr(opt_cfg, "lr", 2e-4)
        betas = tuple(getattr(opt_cfg, "betas", [0.5, 0.999]))

        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        self.criterion = nn.BCELoss()
        self.label_smoothing = getattr(dcfg, "label_smoothing", 0.1)
        self.noise_injection = getattr(dcfg, "noise_injection", 0.05)
        self.latent_dim = latent_dim

        self.fixed_z = torch.randn(64, latent_dim, device=self.device)

        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        self.logger.info(f"Generator params: {g_params:,} | Discriminator params: {d_params:,}")

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        g_total = d_total = d_acc_total = 0.0
        n = 0

        for images, _ in tqdm(dataloader, desc="Train DCGAN", leave=False):
            batch_size = images.size(0)
            images = images.to(self.device)

            # Noise injection for discriminator inputs
            if self.noise_injection > 0:
                images = images + self.noise_injection * torch.randn_like(images)

            real_label = torch.full((batch_size, 1), 1.0 - self.label_smoothing, device=self.device)
            fake_label = torch.full((batch_size, 1), self.label_smoothing, device=self.device)

            # ── Train Discriminator ───────────────────────────────────
            self.opt_d.zero_grad()
            d_real = self.discriminator(images)
            loss_d_real = self.criterion(d_real, real_label)

            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake = self.generator(z).detach()
            if self.noise_injection > 0:
                fake = fake + self.noise_injection * torch.randn_like(fake)
            d_fake = self.discriminator(fake)
            loss_d_fake = self.criterion(d_fake, fake_label)

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                           self.config.training.gradient_clip)
            self.opt_d.step()

            d_acc = ((d_real > 0.5).float().mean() + (d_fake < 0.5).float().mean()) / 2

            # ── Train Generator ───────────────────────────────────────
            self.opt_g.zero_grad()
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake = self.generator(z)
            d_fake_for_g = self.discriminator(fake)
            loss_g = self.criterion(d_fake_for_g, torch.ones(batch_size, 1, device=self.device))
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                                           self.config.training.gradient_clip)
            self.opt_g.step()

            g_total += loss_g.item()
            d_total += loss_d.item()
            d_acc_total += d_acc.item()
            n += 1
            self.global_step += 1

        return {"g_loss": g_total / n, "d_loss": d_total / n, "d_acc": d_acc_total / n}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.generator.eval()
        self.discriminator.eval()
        d_total = 0.0
        n = 0
        for images, _ in dataloader:
            images = images.to(self.device)
            d_real = self.discriminator(images)
            z = torch.randn(images.size(0), self.latent_dim, device=self.device)
            fake = self.generator(z)
            d_fake = self.discriminator(fake)
            d_loss = (self.criterion(d_real, torch.ones_like(d_real)) +
                      self.criterion(d_fake, torch.zeros_like(d_fake))) / 2
            d_total += d_loss.item()
            n += 1
        return {"d_loss": d_total / n}

    @torch.no_grad()
    def generate_samples(self, epoch: int) -> None:
        self.generator.eval()
        samples = self.generator(self.fixed_z)
        grid = vutils.make_grid(samples, nrow=8, normalize=True, padding=2)
        self.logger.log_images("dcgan/generated", samples, epoch)
        vutils.save_image(grid, self.generated_dir / f"dcgan_epoch_{epoch:04d}.png")

    def _save_checkpoint(self, epoch, metrics):
        models = {"generator": self.generator, "discriminator": self.discriminator}
        optims = {"opt_g": self.opt_g, "opt_d": self.opt_d}
        self.ckpt_mgr.save(models, optims, epoch, metrics)
        self.ckpt_mgr.save_best(models, optims, epoch,
                                metrics.get("g_loss", float("inf")), metrics)
