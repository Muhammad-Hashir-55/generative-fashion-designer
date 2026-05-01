"""Lightning module for fast VAE training.

Designed for multi-GPU execution, mixed precision, and readiness-gated
preflight validation before full training.
"""

from __future__ import annotations

import torch
import lightning.pytorch as pl

from src.models.vae import VAE


class LightningVAE(pl.LightningModule):
    """PyTorch Lightning module wrapping the project VAE."""

    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        vae_cfg = config.models.vae
        self.model = VAE(
            in_channels=config.data.channels,
            latent_dim=config.models.latent_dim,
            encoder_channels=vae_cfg.encoder_channels,
            decoder_channels=vae_cfg.decoder_channels,
            image_size=config.data.image_size,
            use_residual=getattr(vae_cfg, "use_residual", True),
            use_pretrained_encoder=getattr(vae_cfg, "use_pretrained_encoder", False),
            freeze_pretrained_encoder=getattr(vae_cfg, "freeze_pretrained_encoder", True),
        )
        self.beta_target = float(getattr(vae_cfg, "beta", 1.0))
        self.beta_anneal_epochs = int(getattr(vae_cfg, "beta_anneal_epochs", 10))

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _get_beta(self) -> float:
        if self.beta_anneal_epochs <= 0:
            return self.beta_target
        progress = min(1.0, float(self.current_epoch + 1) / float(self.beta_anneal_epochs))
        return progress * self.beta_target

    def training_step(self, batch, batch_idx):
        images, _ = batch
        recon, mu, logvar = self.model(images)
        beta = self._get_beta()
        loss, recon_loss, kl_loss = VAE.loss_function(recon, images, mu, logvar, beta)
        self.log_dict(
            {
                "train_loss": loss,
                "train_recon": recon_loss,
                "train_kl": kl_loss,
                "train_beta": torch.tensor(beta, device=self.device),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=images.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        recon, mu, logvar = self.model(images)
        loss, recon_loss, kl_loss = VAE.loss_function(recon, images, mu, logvar, self.beta_target)
        self.log_dict(
            {
                "val_loss": loss,
                "val_recon": recon_loss,
                "val_kl": kl_loss,
            },
            prog_bar=True,
            on_epoch=True,
            batch_size=images.size(0),
        )
        return {"val_loss": loss, "val_recon": recon_loss, "val_kl": kl_loss}

    def configure_optimizers(self):
        opt_cfg = getattr(self.config.training, "vae_optimizer", self.config.training.optimizer)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(getattr(opt_cfg, "lr", 1e-3)),
            betas=tuple(getattr(opt_cfg, "betas", [0.9, 0.999])),
        )
        return optimizer
