"""
VAE Trainer
============
Trains the β-VAE with KL annealing, latent space visualization,
and reconstruction quality tracking.
"""

from __future__ import annotations

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vae import VAE
from src.training.base_trainer import BaseTrainer
from src.training.scheduler import build_scheduler
from src.utils.config import Config


class VAETrainer(BaseTrainer):

    def __init__(self, config: Config, device: torch.device | None = None):
        super().__init__(config, "vae", device)

    def setup(self, **kwargs) -> None:
        vae_cfg = self.config.models.vae
        self.model = VAE(
            in_channels=self.config.data.channels,
            latent_dim=self.config.models.latent_dim,
            encoder_channels=vae_cfg.encoder_channels,
            decoder_channels=vae_cfg.decoder_channels,
            image_size=self.config.data.image_size,
            use_residual=getattr(vae_cfg, "use_residual", True),
            use_pretrained_encoder=getattr(vae_cfg, "use_pretrained_encoder", False),
            freeze_pretrained_encoder=getattr(vae_cfg, "freeze_pretrained_encoder", True),
        ).to(self.device)

        opt_cfg = getattr(self.config.training, "vae_optimizer", self.config.training.optimizer)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(opt_cfg, "lr", 1e-3),
            betas=tuple(getattr(opt_cfg, "betas", [0.9, 0.999])),
        )
        self.scheduler = build_scheduler(self.optimizer, self.config)

        self.beta_target = getattr(vae_cfg, "beta", 1.0)
        self.beta_anneal_epochs = getattr(vae_cfg, "beta_anneal_epochs", 10)

        # Fixed noise for consistent sample visualization
        self.fixed_z = torch.randn(64, self.config.models.latent_dim, device=self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"VAE parameters: {n_params:,}")

    def _get_beta(self) -> float:
        if self.beta_anneal_epochs <= 0:
            return self.beta_target
        return min(1.0, self.current_epoch / self.beta_anneal_epochs) * self.beta_target

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = total_recon = total_kl = 0.0
        beta = self._get_beta()
        n = 0

        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Train", leave=False)):
            images = images.to(self.device)
            with self.autocast():
                recon, mu, logvar = self.model(images)
                loss, recon_loss, kl_loss = VAE.loss_function(recon, images, mu, logvar, beta)

            self.optimizer.zero_grad()
            self.backward_step(
                loss,
                self.optimizer,
                clip_params=self.model.parameters(),
                clip_value=self.config.training.gradient_clip,
            )

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n += 1
            self.global_step += 1

        if self.scheduler:
            self.scheduler.step()

        return {"loss": total_loss / n, "recon": total_recon / n,
                "kl": total_kl / n, "beta": beta}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = total_recon = total_kl = 0.0
        n = 0
        for images, _ in dataloader:
            images = images.to(self.device)
            with self.autocast(enabled=False):
                recon, mu, logvar = self.model(images)
                loss, recon_loss, kl_loss = VAE.loss_function(recon, images, mu, logvar, self.beta_target)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n += 1
        return {"loss": total_loss / n, "recon": total_recon / n, "kl": total_kl / n}

    @torch.no_grad()
    def generate_samples(self, epoch: int) -> None:
        self.model.eval()
        samples = self.model.decode(self.fixed_z)
        grid = vutils.make_grid(samples, nrow=8, normalize=True, padding=2)
        self.logger.log_images("vae/generated", samples, epoch)
        vutils.save_image(grid, self.generated_dir / f"vae_epoch_{epoch:04d}.png")

    def _save_checkpoint(self, epoch, metrics):
        self.ckpt_mgr.save(self.model, self.optimizer, epoch, metrics)
        self.ckpt_mgr.save_best(self.model, self.optimizer, epoch,
                                metrics.get("loss", float("inf")), metrics)
