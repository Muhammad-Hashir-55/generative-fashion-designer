"""
Dataset Visualization Script
==============================
Visualize Describable Textures Dataset (DTD) samples, class distributions,
and augmentation effects for the Generative Fashion Designer project.

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
import numpy as np
from torchvision.datasets import DTD

from src.utils.config import load_config
from src.data.dataset import FashionDesignerDataset, get_class_names
from src.data.augmentation import TrainAugmentation, EvalAugmentation, GANAugmentation


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize an RGB tensor for plotting.

    Supports two common normalizations:
    - GAN-style: tensor in [-1, 1] -> map to [0, 1]
    - ImageNet-style: (x - mean) / std -> invert to [0, 1]
    """
    t = tensor.detach().cpu()
    img = t.squeeze().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    # Heuristic: if values are already in [-1, 1], assume GAN normalization
    if t.min() >= -1.01 and t.max() <= 1.01:
        img = (img + 1.0) / 2.0
        return np.clip(img, 0, 1)

    # Otherwise assume ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std) + mean
    return np.clip(img, 0, 1)


def main():
    config = load_config()
    output_dir = Path(getattr(config.paths, "evaluation_dir", "./outputs/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    print("Loading Describable Textures Dataset (DTD)...")
    raw_dataset = FashionDesignerDataset(
        root=getattr(config.data, "data_dir", "./data"),
        split="train",
        transform=EvalAugmentation(config.data.image_size),
        download=True,
    )
    classes = get_class_names(getattr(config.data, "data_dir", "./data"))
    print(f"  Training set size: {len(raw_dataset):,}")
    print(f"  Total Classes: {len(classes)}")

    # ── 1. Raw Dataset Samples ────────────────────────────────────────────
    # Plot 10 rows for 10 selected pattern-like classes
    selected_classes = [
        "paisley", "woven", "interlaced", "knitted", "zigzag",
        "banded", "chequered", "grid", "polka-dotted", "marbled"
    ]
    # Filter to only existing classes to avoid errors
    selected_classes = [c for c in selected_classes if c in classes]
    
    num_classes_to_plot = min(len(selected_classes), 10)
    fig, axes = plt.subplots(num_classes_to_plot, 10, figsize=(15, 1.5 * num_classes_to_plot))
    
    class_indices = {classes.index(c): [] for c in selected_classes[:num_classes_to_plot]}
    
    for idx in range(len(raw_dataset)):
        _, label = raw_dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label in class_indices and len(class_indices[label]) < 10:
            class_indices[label].append(idx)
        if all(len(v) >= 10 for v in class_indices.values()):
            break

    for row, c_name in enumerate(selected_classes[:num_classes_to_plot]):
        c_idx = classes.index(c_name)
        for col in range(10):
            if col < len(class_indices[c_idx]):
                idx = class_indices[c_idx][col]
                img, _ = raw_dataset[idx]
                img_np = denormalize(img)
                axes[row][col].imshow(img_np)
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(c_name, fontsize=10, rotation=0, labelpad=50, va="center")

    fig.suptitle("Describable Textures Dataset (DTD) — Selected Pattern Classes", 
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "dataset_samples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved class samples to {path}")

    # ── 2. Class Distribution ─────────────────────────────────────────────
    # Count distribution using raw targets
    targets = [raw_dataset.dataset._labels[i] for i in range(len(raw_dataset))]
    counts = [targets.count(i) for i in range(len(classes))]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(classes, counts, color="#4ECDC4", edgecolor="white", linewidth=0.5)
    ax.set_title("DTD Class Distribution (Train Split)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    plt.tight_layout()
    path = output_dir / "class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved class distribution to {path}")

    # ── 3. Augmentation Comparison ────────────────────────────────────────
    print("  Generating augmentation comparison...")
    raw = DTD(root=getattr(config.data, "data_dir", "./data"), split="train", download=True)
    sample_img = raw[0][0]  # PIL Image

    aug_pipelines = {
        "Original": EvalAugmentation(config.data.image_size),
        "Train Aug": TrainAugmentation(config.data.image_size),
        "GAN Aug": GANAugmentation(config.data.image_size),
    }

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for row, (name, aug) in enumerate(aug_pipelines.items()):
        for col in range(8):
            img = aug(sample_img)
            img_np = denormalize(img)
            axes[row][col].imshow(img_np)
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(name, fontsize=11, rotation=0, labelpad=50, va="center")

    fig.suptitle("RGB Augmentation Pipeline Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "augmentation_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved augmentation comparison to {path}")

    print("\n  Dataset visualization complete!")
    print(f"  All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
