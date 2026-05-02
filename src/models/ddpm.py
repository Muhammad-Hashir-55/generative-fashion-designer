"""
Denoising Diffusion Probabilistic Model (DDPM)
================================================
Implements the UNet architecture and the noise scheduling
for a DDPM based on Ho et al., 2020.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import ResidualBlock, SelfAttention

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.transform = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        h = self.conv1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.transform(h)
        h = self.conv2(h)
        return h + self.res_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=128, channel_mults=(1, 2, 4, 8)):
        super().__init__()
        self.time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        channels = base_channels
        now_channels = base_channels
        feat_dims = [now_channels]
        
        for i, mult in enumerate(channel_mults):
            out_dim = base_channels * mult
            is_last = i == len(channel_mults) - 1
            self.downs.append(nn.ModuleList([
                Block(now_channels, out_dim, self.time_dim),
                Block(out_dim, out_dim, self.time_dim),
                nn.Conv2d(out_dim, out_dim, 4, 2, 1) if not is_last else nn.Conv2d(out_dim, out_dim, 3, padding=1)
            ]))
            now_channels = out_dim
            feat_dims.append(now_channels)

        self.mid_block1 = Block(now_channels, now_channels, self.time_dim)
        self.mid_attn = SelfAttention(now_channels)
        self.mid_block2 = Block(now_channels, now_channels, self.time_dim)

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_dim = base_channels * mult
            is_last = i == 0
            self.ups.append(nn.ModuleList([
                Block(now_channels + feat_dims.pop(), out_dim, self.time_dim),
                Block(out_dim, out_dim, self.time_dim),
                nn.ConvTranspose2d(out_dim, out_dim, 4, 2, 1) if not is_last else nn.Conv2d(out_dim, out_dim, 3, padding=1)
            ]))
            now_channels = out_dim

        self.final_res_block = Block(now_channels + feat_dims.pop(), base_channels, self.time_dim)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.init_conv(x)
        
        skips = [x]
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)

        x = torch.cat((x, skips.pop()), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

class DDPM(nn.Module):
    def __init__(self, unet: UNet, timesteps: int = 1000):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps

        beta = torch.linspace(0.0001, 0.02, timesteps)
        alpha = 1. - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)

    def forward(self, x):
        device = x.device
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
        noise = torch.randn_like(x)

        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        x_noisy = torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * noise

        pred_noise = self.unet(x_noisy, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples, image_size=64, channels=3, device="cuda"):
        x = torch.randn((num_samples, channels, image_size, image_size), device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t)
            
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
            beta_t = self.beta[t].view(-1, 1, 1, 1)

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise) + torch.sqrt(beta_t) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        return x
