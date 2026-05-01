"""
Dataset Visualization Script
==============================
Visualize Fashion-MNIST samples, class distributions, and augmentation effects.

Usage:
    python scripts/visualize_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.utils.config import load_config
from src.data.dataset import FashionDesignerDataset, FASHION_CLASSES
from src.data.augmentation import TrainAugmentation, EvalAugmentation, GANAugmentation


def main():
    config = load_config()
    output_dir = Path(getattr(config.paths, "evaluation_dir", "./outputs/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")

    # ── 1. Raw Dataset Samples ────────────────────────────────────────────
    print("Loading Fashion-MNIST dataset...")
    raw_dataset = FashionDesignerDataset(
        root=getattr(config.data, "data_dir", "./data"),
        train=True,
        transform=EvalAugmentation(config.data.image_size),
        download=True,
    )
    print(f"  Training set size: {len(raw_dataset):,}")
    print(f"  Classes: {FASHION_CLASSES}")

    # Plot 10 × 10 grid (one row per class)
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    class_indices = {i: [] for i in range(10)}
    for idx in range(len(raw_dataset)):
        _, label = raw_dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if len(class_indices[label]) < 10:
            class_indices[label].append(idx)
        if all(len(v) >= 10 for v in class_indices.values()):
            break

    for row in range(10):
        for col in range(10):
            idx = class_indices[row][col]
            img, _ = raw_dataset[idx]
            img_np = img.squeeze().numpy()
            img_np = (img_np + 1) / 2  # denormalize
            axes[row][col].imshow(img_np, cmap="gray", vmin=0, vmax=1)
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(FASHION_CLASSES[row], fontsize=9,
                                 rotation=0, labelpad=70, va="center")

    fig.suptitle("Fashion-MNIST — All 10 Classes", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "dataset_samples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved class samples to {path}")

    # ── 2. Class Distribution ─────────────────────────────────────────────
    targets = raw_dataset.targets
    counts = [targets.count(i) for i in range(10)]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.get_cmap("viridis", 10)
    bars = ax.bar(FASHION_CLASSES, counts, color=[colors(i) for i in range(10)],
                  edgecolor="white", linewidth=0.5)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", fontsize=9)
    plt.tight_layout()
    path = output_dir / "class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved class distribution to {path}")

    # ── 3. Augmentation Comparison ────────────────────────────────────────
    print("  Generating augmentation comparison...")
    from torchvision.datasets import FashionMNIST
    raw = FashionMNIST(root=getattr(config.data, "data_dir", "./data"),
                       train=True, download=True)
    sample_img = raw[0][0]  # PIL Image

    aug_pipelines = {
        "Original": EvalAugmentation(config.data.image_size),
        "Train Aug": TrainAugmentation(config.data.image_size),
        "GAN Aug": GANAugmentation(config.data.image_size),
    }

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for row, (name, aug) in enumerate(aug_pipelines.items()):
        for col in range(8):
            img = aug(sample_img).squeeze().numpy()
            img = (img + 1) / 2
            axes[row][col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(name, fontsize=11, rotation=0, labelpad=60, va="center")

    fig.suptitle("Augmentation Pipeline Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "augmentation_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved augmentation comparison to {path}")

    print("\n  Dataset visualization complete!")
    print(f"  All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
