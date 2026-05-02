import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BATCH_SIZE = 32
IMAGE_SIZE = 64
CHANNELS = 3
EPOCHS = 100
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data"
OUTPUT_DIR = "./ddpm_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset and DataLoader
# We use the Describable Textures Dataset (DTD)
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
])

print("Downloading/Loading DTD Dataset...")
dataset = torchvision.datasets.DTD(root=DATA_DIR, split='train', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
print(f"Dataset loaded: {len(dataset)} images.")

# Model Architecture (Stable Version)
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

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

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
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, channel_mults=(1, 2, 4, 8)):
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
        
        # CRITICAL FIX for NaN: Float16 precision loss causes (1 - alpha_hat_t) to evaluate to 
        # exact 0.0 or negative, causing torch.sqrt() to return NaN. 
        # We enforce a strict minimum variance clip.
        alpha_hat_t = torch.clamp(alpha_hat_t, max=0.9999)
        one_minus_alpha_hat_t = torch.clamp(1.0 - alpha_hat_t, min=1e-5)
        
        x_noisy = torch.sqrt(alpha_hat_t) * x + torch.sqrt(one_minus_alpha_hat_t) * noise

        pred_noise = self.unet(x_noisy, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples, image_size=64, channels=3, device="cuda"):
        x = torch.randn((num_samples, channels, image_size, image_size), device=device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", leave=False, total=self.timesteps):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t)
            
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
            beta_t = self.beta[t].view(-1, 1, 1, 1)
            
            # CRITICAL FIX for NaN sampling:
            one_minus_alpha_hat_t = torch.clamp(1.0 - alpha_hat_t, min=1e-5)

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(one_minus_alpha_hat_t)) * pred_noise) + torch.sqrt(beta_t) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        return x

# Training Loop with NaN Prevention
unet = UNet(in_channels=CHANNELS, out_channels=CHANNELS, base_channels=64).to(DEVICE)
model = DDPM(unet, timesteps=1000).to(DEVICE)

# AdamW is significantly more stable for diffusion UNets than plain Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Mixed precision scaler (we disable it internally if we hit NaN, but grad clipping usually prevents it)
scaler = torch.cuda.amp.GradScaler(enabled=True)

print(f"Starting training on {DEVICE}...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        images, _ = batch
        images = images.to(DEVICE)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=True):
            loss = model(images)
            
        # CRITICAL FIX: explicit NaN/Inf detection. Skip batch immediately rather than corrupting network.
        if not math.isfinite(loss.item()):
            print(f"\nWARNING: NaN loss detected at epoch {epoch}. Skipping batch.")
            continue
            
        scaler.scale(loss).backward()
        
        # CRITICAL FIX: Gradient Clipping is strictly required for stable DDPM training under AMP.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        valid_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    avg_loss = total_loss / max(1, valid_batches)
    print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")
    
    # Save Checkpoint safely
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, os.path.join(OUTPUT_DIR, "ddpm_latest.pt"))
    
    # Generate and Save Samples every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        print(f"Generating samples for epoch {epoch}...")
        with torch.no_grad():
            samples = model.sample(num_samples=16, image_size=IMAGE_SIZE, channels=CHANNELS, device=DEVICE)
            grid = vutils.make_grid(samples, nrow=4, padding=2, normalize=False)
            
            # Save to disk
            sample_path = os.path.join(OUTPUT_DIR, f"sample_epoch_{epoch:03d}.png")
            vutils.save_image(samples, sample_path, nrow=4, padding=2, normalize=False)
            
            # Display in notebook
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(f"Epoch {epoch} Samples")
            plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
            plt.show()

print("Training Complete!")

