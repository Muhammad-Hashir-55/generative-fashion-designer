"""
Result Visualizer
==================
Rich matplotlib visualizations for training results: loss curves,
sample grids, latent space t-SNE, interpolation walks, and more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE


class ResultVisualizer:
    """Publication-quality visualization for generative model outputs."""

    def __init__(self, output_dir: str = "./outputs/evaluation",
                 dpi: int = 150, figsize: tuple = (12, 8)) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        plt.style.use("dark_background")

    def plot_training_curves(self, history: dict[str, list[float]],
                             model_name: str) -> Path:
        """Plot training loss curves."""
        fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 5))
        if len(history) == 1:
            axes = [axes]

        colors = ["#00D4FF", "#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3"]
        for ax, (key, vals), color in zip(axes, history.items(), colors):
            ax.plot(vals, color=color, linewidth=2, alpha=0.9)
            ax.set_title(key.replace("_", " ").title(), fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Value", fontsize=11)
            ax.grid(True, alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(f"{model_name} Training Curves", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = self.output_dir / f"{model_name}_training_curves.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def plot_gan_loss_comparison(self, g_losses: list[float],
                                 d_losses: list[float],
                                 model_name: str) -> Path:
        """Plot Generator vs Discriminator loss curves."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(g_losses, color="#FF6B6B", linewidth=2, label="Generator", alpha=0.85)
        ax.plot(d_losses, color="#4ECDC4", linewidth=2, label="Discriminator", alpha=0.85)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"{model_name} — G vs D Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, framealpha=0.3)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        path = self.output_dir / f"{model_name}_gd_loss.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def plot_sample_grid(self, images: torch.Tensor, title: str,
                         filename: str, nrow: int = 8) -> Path:
        """Plot a grid of generated images."""
        n = min(images.size(0), nrow * nrow)
        imgs = images[:n].cpu()
        if imgs.min() < 0:
            imgs = (imgs + 1) / 2

        fig, axes = plt.subplots(nrow, nrow, figsize=(nrow * 1.5, nrow * 1.5))
        for i in range(nrow):
            for j in range(nrow):
                idx = i * nrow + j
                ax = axes[i][j]
                if idx < n:
                    img = imgs[idx].squeeze().numpy()
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = self.output_dir / f"{filename}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    @torch.no_grad()
    def plot_latent_tsne(self, model, dataloader, model_name: str,
                          class_names: list[str] | None = None,
                          n_samples: int = 2000) -> Path:
        """t-SNE visualization of the VAE latent space."""
        model.eval()
        latents, labels_list = [], []
        for imgs, lbls in dataloader:
            if sum(len(l) for l in latents) >= n_samples:
                break
            mu, _ = model.encode(imgs.to(next(model.parameters()).device))
            latents.append(mu.cpu().numpy())
            labels_list.append(lbls.numpy())

        z = np.concatenate(latents)[:n_samples]
        y = np.concatenate(labels_list)[:n_samples]

        print(f"  Running t-SNE on {len(z)} latent vectors...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        z_2d = tsne.fit_transform(z)

        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = plt.cm.get_cmap("tab10", 10)
        for c in range(10):
            mask = y == c
            label = class_names[c] if class_names else str(c)
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[cmap(c)],
                       s=8, alpha=0.6, label=label)

        ax.legend(fontsize=9, markerscale=3, framealpha=0.3,
                  loc="upper right")
        ax.set_title(f"{model_name} Latent Space (t-SNE)",
                     fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.15)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        path = self.output_dir / f"{model_name}_tsne.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    @torch.no_grad()
    def plot_interpolation(self, model, z1: torch.Tensor, z2: torch.Tensor,
                            steps: int = 10, model_name: str = "model") -> Path:
        """Latent space linear interpolation between two points."""
        model.eval()
        device = next(model.parameters()).device
        alphas = torch.linspace(0, 1, steps)
        interp_images = []
        for a in alphas:
            z = (1 - a) * z1 + a * z2
            img = model.decode(z.unsqueeze(0).to(device))
            interp_images.append(img.cpu().squeeze())

        fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2))
        for i, (ax, img) in enumerate(zip(axes, interp_images)):
            img_np = img.squeeze().numpy()
            if img_np.min() < 0:
                img_np = (img_np + 1) / 2
            ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title(f"α={alphas[i]:.1f}", fontsize=8)

        fig.suptitle(f"{model_name} Latent Interpolation",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = self.output_dir / f"{model_name}_interpolation.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    def plot_reconstruction_comparison(self, originals: torch.Tensor,
                                        reconstructions: torch.Tensor,
                                        model_name: str, n: int = 8) -> Path:
        """Side-by-side original vs reconstruction."""
        fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
        for i in range(n):
            orig = originals[i].cpu().squeeze().numpy()
            recon = reconstructions[i].cpu().squeeze().numpy()
            if orig.min() < 0:
                orig = (orig + 1) / 2
            if recon.min() < 0:
                recon = (recon + 1) / 2
            axes[0, i].imshow(orig, cmap="gray", vmin=0, vmax=1)
            axes[0, i].axis("off")
            axes[1, i].imshow(recon, cmap="gray", vmin=0, vmax=1)
            axes[1, i].axis("off")

        axes[0, 0].set_ylabel("Original", fontsize=11, rotation=0, labelpad=50)
        axes[1, 0].set_ylabel("Recon", fontsize=11, rotation=0, labelpad=50)
        fig.suptitle(f"{model_name} Reconstruction", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = self.output_dir / f"{model_name}_reconstruction.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
