"""
WGAN-GP Trainer
================
Wasserstein training with gradient penalty, 5:1 critic-to-generator
ratio, and Earth Mover's Distance monitoring.
"""

from __future__ import annotations

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.wgan_gp import WGANGenerator, WGANCritic, compute_gradient_penalty
from src.training.base_trainer import BaseTrainer
from src.utils.config import Config


class WGANTrainer(BaseTrainer):

    def __init__(self, config: Config, device: torch.device | None = None):
        super().__init__(config, "wgan_gp", device)

    def setup(self, **kwargs) -> None:
        wcfg = self.config.models.wgan_gp
        latent_dim = self.config.models.latent_dim

        self.generator = WGANGenerator(
            latent_dim=latent_dim, channels=wcfg.g_channels,
            out_channels=self.config.data.channels,
            image_size=self.config.data.image_size,
        ).to(self.device)

        self.critic = WGANCritic(
            in_channels=self.config.data.channels,
            channels=wcfg.critic_channels,
            image_size=self.config.data.image_size,
        ).to(self.device)

        opt_cfg = self.config.training.optimizer
        lr = getattr(opt_cfg, "lr", 2e-4)
        betas = tuple(getattr(opt_cfg, "betas", [0.5, 0.999]))

        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas)

        self.lambda_gp = getattr(wcfg, "lambda_gp", 10.0)
        self.n_critic = getattr(wcfg, "n_critic", 5)
        self.latent_dim = latent_dim
        self.fixed_z = torch.randn(64, latent_dim, device=self.device)

        g_p = sum(p.numel() for p in self.generator.parameters())
        c_p = sum(p.numel() for p in self.critic.parameters())
        self.logger.info(f"Generator: {g_p:,} | Critic: {c_p:,}")
        # WGAN-GP is sensitive to AMP scaler ordering on alternating optimizer steps.
        # Keep the model on GPU, but run this trainer in full precision.
        self.amp_enabled = False
        self.scaler = None

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.generator.train()
        self.critic.train()
        g_total = c_total = gp_total = w_dist_total = 0.0
        n_g = n_c = 0

        data_iter = iter(dataloader)
        num_batches = len(dataloader)
        batch_idx = 0

        while batch_idx < num_batches:
            # ── Train Critic n_critic times ──────────────────────
            for _ in range(self.n_critic):
                try:
                    images, _ = next(data_iter)
                except StopIteration:
                    break
                batch_idx += 1
                images = images.to(self.device)
                bs = images.size(0)

                z = torch.randn(bs, self.latent_dim, device=self.device)
                with self.autocast():
                    fake = self.generator(z).detach()

                with self.autocast():
                    c_real = self.critic(images).mean()
                    c_fake = self.critic(fake).mean()
                gp = compute_gradient_penalty(
                    self.critic, images, fake, self.device, self.lambda_gp)

                c_loss = c_fake - c_real + gp
                self.opt_c.zero_grad()
                self.backward_step(
                    c_loss,
                    self.opt_c,
                    clip_params=self.critic.parameters(),
                    clip_value=self.config.training.gradient_clip,
                    scaler_update=False,
                )

                w_dist = (c_real - c_fake).item()
                c_total += c_loss.item()
                gp_total += gp.item()
                w_dist_total += w_dist
                n_c += 1

            # ── Train Generator ──────────────────────────────────
            z = torch.randn(images.size(0), self.latent_dim, device=self.device)
            with self.autocast():
                fake = self.generator(z)
                g_loss = -self.critic(fake).mean()

            self.opt_g.zero_grad()
            self.backward_step(
                g_loss,
                self.opt_g,
                clip_params=self.generator.parameters(),
                clip_value=self.config.training.gradient_clip,
                scaler_update=True,
            )

            g_total += g_loss.item()
            n_g += 1
            self.global_step += 1

        return {
            "g_loss": g_total / max(n_g, 1),
            "c_loss": c_total / max(n_c, 1),
            "gp": gp_total / max(n_c, 1),
            "w_dist": w_dist_total / max(n_c, 1),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.critic.eval()
        self.generator.eval()
        w_total = 0.0
        n = 0
        for images, _ in dataloader:
            images = images.to(self.device)
            with self.autocast(enabled=False):
                z = torch.randn(images.size(0), self.latent_dim, device=self.device)
                fake = self.generator(z)
                w = (self.critic(images).mean() - self.critic(fake).mean()).item()
            w_total += w
            n += 1
        return {"w_dist": w_total / max(n, 1)}

    @torch.no_grad()
    def generate_samples(self, epoch: int) -> None:
        self.generator.eval()
        samples = self.generator(self.fixed_z)
        grid = vutils.make_grid(samples, nrow=8, normalize=True, padding=2)
        self.logger.log_images("wgan_gp/generated", samples, epoch)
        vutils.save_image(grid, self.generated_dir / f"wgan_gp_epoch_{epoch:04d}.png")

    def _save_checkpoint(self, epoch, metrics):
        models = {"generator": self.generator, "critic": self.critic}
        optims = {"opt_g": self.opt_g, "opt_c": self.opt_c}
        self.ckpt_mgr.save(models, optims, epoch, metrics)
        self.ckpt_mgr.save_best(models, optims, epoch,
                                -metrics.get("w_dist", 0), metrics,
                                lower_is_better=True)
