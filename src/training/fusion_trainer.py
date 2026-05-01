"""
CVAE-GAN Fusion Trainer
========================
Triple-loss training: alternating encoder-decoder and discriminator updates.
"""

from __future__ import annotations

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.fusion_generator import CVAEGANFusion
from src.training.base_trainer import BaseTrainer
from src.utils.config import Config


class FusionTrainer(BaseTrainer):

    def __init__(self, config: Config, device: torch.device | None = None):
        super().__init__(config, "fusion", device)

    def setup(self, **kwargs) -> None:
        fcfg = self.config.models.fusion
        latent_dim = self.config.models.latent_dim

        self.model = CVAEGANFusion(
            in_channels=self.config.data.channels,
            latent_dim=latent_dim,
            encoder_channels=getattr(fcfg, "encoder_channels", None),
            decoder_channels=getattr(fcfg, "decoder_channels", None),
            d_channels=getattr(fcfg, "d_channels", None),
            image_size=self.config.data.image_size,
        ).to(self.device)

        opt_cfg = self.config.training.optimizer
        lr = getattr(opt_cfg, "lr", 2e-4)
        betas = tuple(getattr(opt_cfg, "betas", [0.5, 0.999]))

        # Separate optimizers for encoder+decoder vs discriminator
        enc_dec_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        self.opt_enc_dec = torch.optim.Adam(enc_dec_params, lr=lr, betas=betas)
        self.opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=betas)

        self.recon_w = getattr(fcfg, "recon_weight", 10.0)
        self.kl_w = getattr(fcfg, "kl_weight", 1.0)
        self.adv_w = getattr(fcfg, "adv_weight", 1.0)
        self.latent_dim = latent_dim

        self.fixed_z = torch.randn(64, latent_dim, device=self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"CVAE-GAN Fusion parameters: {n_params:,}")

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals = {"enc_dec": 0, "d": 0, "recon": 0, "kl": 0}
        n = 0

        for images, _ in tqdm(dataloader, desc="Train Fusion", leave=False):
            images = images.to(self.device)

            with self.autocast():
                # Forward pass for discriminator update
                recon, mu, logvar = self.model(images)
                d_real = self.model.discriminator(images)
                d_fake = self.model.discriminator(recon.detach())

                losses = CVAEGANFusion.compute_losses(
                    recon, images, mu, logvar, d_real, d_fake,
                    self.recon_w, self.kl_w, self.adv_w)

            # ── Update Discriminator ──
            self.opt_d.zero_grad()
            self.backward_step(
                losses["d_loss"],
                self.opt_d,
                clip_params=self.model.discriminator.parameters(),
                clip_value=self.config.training.gradient_clip,
                scaler_update=False,
            )

            # ── Update Encoder + Decoder ──
            # Recompute forward with gradients for generator branch.
            with self.autocast():
                recon_g, mu_g, logvar_g = self.model(images)
                d_real_g = self.model.discriminator(images).detach()
                d_fake_g = self.model.discriminator(recon_g)
                losses_g = CVAEGANFusion.compute_losses(
                    recon_g, images, mu_g, logvar_g, d_real_g, d_fake_g,
                    self.recon_w, self.kl_w, self.adv_w)

            self.opt_enc_dec.zero_grad()
            self.backward_step(
                losses_g["enc_dec_loss"],
                self.opt_enc_dec,
                clip_params=list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                clip_value=self.config.training.gradient_clip,
                scaler_update=True,
            )

            totals["enc_dec"] += losses_g["enc_dec_loss"].item()
            totals["d"] += losses["d_loss"].item()
            totals["recon"] += losses["recon_loss"].item()
            totals["kl"] += losses["kl_loss"].item()
            n += 1
            self.global_step += 1

        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        recon_total = kl_total = 0.0
        n = 0
        for images, _ in dataloader:
            images = images.to(self.device)
            with self.autocast(enabled=False):
                recon, mu, logvar = self.model(images)
                target_01 = (images + 1.0) / 2.0
                recon_scaled = (recon.float() + 1.0) / 2.0
                # Handle NaNs and clamp to [0, 1] to avoid CUDA device-side asserts
                if not torch.isfinite(recon_scaled).all():
                    recon_scaled = torch.nan_to_num(recon_scaled, nan=0.0, posinf=1.0, neginf=0.0)
                recon_scaled = torch.clamp(recon_scaled, 0.0, 1.0)
                target_01 = torch.clamp(target_01.float(), 0.0, 1.0)
                
                recon_loss = torch.nn.functional.binary_cross_entropy(
                    recon_scaled, target_01, reduction="sum") / images.size(0)
                kl_loss = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
            recon_total += recon_loss.item()
            kl_total += kl_loss.item()
            n += 1
        return {"recon": recon_total / n, "kl": kl_total / n}

    @torch.no_grad()
    def generate_samples(self, epoch: int) -> None:
        self.model.eval()
        samples = self.model.generate(64, self.device)
        grid = vutils.make_grid(samples, nrow=8, normalize=True, padding=2)
        self.logger.log_images("fusion/generated", samples, epoch)
        vutils.save_image(grid, self.generated_dir / f"fusion_epoch_{epoch:04d}.png")

    def _save_checkpoint(self, epoch, metrics):
        models = {"encoder": self.model.encoder, "decoder": self.model.decoder,
                  "discriminator": self.model.discriminator}
        optims = {"opt_enc_dec": self.opt_enc_dec, "opt_d": self.opt_d}
        self.ckpt_mgr.save(models, optims, epoch, metrics)
        self.ckpt_mgr.save_best(models, optims, epoch,
                                metrics.get("recon", float("inf")), metrics)
