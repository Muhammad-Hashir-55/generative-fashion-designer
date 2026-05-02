"""
Latent Diffusion Transformer (Latent DiT)
=========================================
Operates on the 8x8 compressed latent space provided by the VAE.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal Positional Embeddings for timestep encoding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AdaLN(nn.Module):
    """Adaptive Layer Normalization (injects time-step info into Transformer blocks)"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, 2 * dim)
        
    def forward(self, x, c):
        shift, scale = self.linear(c).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = AdaLN(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = AdaLN(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x, c):
        nx = self.norm1(x, c)
        x = x + self.attn(nx, nx, nx)[0]
        nx = self.norm2(x, c)
        x = x + self.mlp(nx)
        return x

class LatentDiT(nn.Module):
    """Diffusion Transformer operating on the 8x8 compressed latent space."""
    def __init__(self, in_channels=4, embed_dim=256, depth=6, n_heads=8, latent_size=8):
        super().__init__()
        # Linear patch embedding (treats each 1x1 latent pixel as a token)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, latent_size * latent_size, embed_dim))
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.blocks = nn.ModuleList([DiTBlock(embed_dim, n_heads) for _ in range(depth)])
        self.norm_out = AdaLN(embed_dim)
        self.head = nn.Linear(embed_dim, in_channels)

    def forward(self, x, time):
        B, C, H, W = x.shape
        
        # Flatten 8x8 grid into sequence of 64 tokens
        x = self.patch_embed(x).flatten(2).transpose(1, 2) # [B, 64, D]
        x = x + self.pos_embed
        
        c = self.time_mlp(time)
        for block in self.blocks:
            x = block(x, c)
            
        x = self.norm_out(x, c)
        x = self.head(x) # [B, 64, 4]
        
        # Reshape back to 8x8 grid
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class StableDDPM(nn.Module):
    def __init__(self, model: LatentDiT, timesteps: int = 1000):
        super().__init__()
        self.model = model
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
        
        # CRITICAL FIX for NaN issues
        alpha_hat_t = torch.clamp(alpha_hat_t, max=0.9999)
        one_minus_alpha_hat_t = torch.clamp(1.0 - alpha_hat_t, min=1e-5)
        
        x_noisy = torch.sqrt(alpha_hat_t) * x + torch.sqrt(one_minus_alpha_hat_t) * noise
        pred_noise = self.model(x_noisy, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples, vae, latent_size=8, channels=4, device="cpu", scale_factor=0.18215):
        """Generates samples by running diffusion in latent space, then decoding via VAE."""
        # 1. Start with pure noise in LATENT space (8x8x4)
        x = torch.randn((num_samples, channels, latent_size, latent_size), device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            pred_noise = self.model(x, t)
            
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
            beta_t = self.beta[t].view(-1, 1, 1, 1)
            
            one_minus_alpha_hat_t = torch.clamp(1.0 - alpha_hat_t, min=1e-5)

            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(one_minus_alpha_hat_t)) * pred_noise) + torch.sqrt(beta_t) * noise

        # 2. Decode the latents back into 64x64 pixel images using the VAE
        x = x / scale_factor
        decoded_images = vae.decode(x)
        # Handle dict or obj from Diffusers VAE, or our custom VAE output
        if hasattr(decoded_images, 'sample'):
            decoded_images = decoded_images.sample
            
        return decoded_images
