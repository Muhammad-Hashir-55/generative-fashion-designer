"""
Trainer for DDPM
=================
Implements the training loop for the Denoising Diffusion Probabilistic Model.
"""

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from src.training.base_trainer import BaseTrainer
from src.models.ddpm import UNet, DDPM
from src.utils.checkpoint import CheckpointManager

class DDPMTrainer(BaseTrainer):
    def __init__(self, config, device):
        super().__init__(config, "ddpm", device)
        
        self.unet = UNet(
            in_channels=config.data.channels,
            out_channels=config.data.channels,
            base_channels=64,
            channel_mults=(1, 2, 4, 8)
        ).to(device)
        
        self.model = DDPM(self.unet, timesteps=1000).to(device)
        
        opt_cfg = config.training.optimizer
        self.optimizer = torch.optim.Adam(
            self.unet.parameters(),
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay
        )

    def setup(self):
        # Additional setup if needed (schedulers)
        pass

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            images, _ = batch
            images = images.to(self.device)
            # GAN normalization is [-1, 1]
            
            with self.autocast():
                loss = self.model(images)
            
            self.optimizer.zero_grad(set_to_none=True)
            self.backward_step(loss, self.optimizer, clip_params=self.unet.parameters(), clip_value=self.config.training.gradient_clip)
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            self.global_step += 1
            if self.global_step % self.config.training.log_every == 0:
                self.logger.log_scalar("train/batch_loss", loss.item(), self.global_step)
        
        return {"loss": total_loss / len(dataloader)}

    def validate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                images, _ = batch
                images = images.to(self.device)
                loss = self.model(images)
                total_loss += loss.item()
                
        return {"loss": total_loss / len(dataloader)}

    def generate_samples(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples=16, image_size=self.config.data.image_size, channels=self.config.data.channels, device=self.device)
            grid = vutils.make_grid(samples, nrow=4, padding=2, normalize=False)
            self.logger.log_images("samples", grid, epoch)
            vutils.save_image(
                samples,
                self.generated_dir / f"ddpm_epoch_{epoch:04d}.png",
                nrow=4, padding=2, normalize=False
            )

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        self.ckpt_mgr.save(state, metrics["loss"], epoch, metric_mode="min")
