"""
Fashion Designer Dataset
=========================
Custom wrapper around ``torchvision.datasets.FashionMNIST`` with
stratified train/val splitting, rich metadata, and label utilities.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset, Subset
import torchvision.datasets as tv_datasets
import numpy as np


# ─── Class Metadata ──────────────────────────────────────────────────────────
FASHION_CLASSES: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

NUM_CLASSES: int = len(FASHION_CLASSES)


def label_to_name(label: int) -> str:
    """Convert integer label to human-readable class name."""
    return FASHION_CLASSES[label]


def name_to_label(name: str) -> int:
    """Convert class name to integer label (case-insensitive)."""
    lower_map = {c.lower(): i for i, c in enumerate(FASHION_CLASSES)}
    key = name.strip().lower()
    if key not in lower_map:
        raise ValueError(
            f"Unknown class '{name}'. Valid: {FASHION_CLASSES}"
        )
    return lower_map[key]


class FashionDesignerDataset(Dataset):
    """Fashion-MNIST wrapper with stratified split support.

    Parameters
    ----------
    root : str
        Download / cache directory.
    train : bool
        If True, load the 60 000-image training split.
    transform : callable, optional
        Torchvision-style transform applied to each PIL image.
    download : bool
        Download if not already cached.
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Any = None,
        download: bool = True,
    ) -> None:
        self.inner = tv_datasets.FashionMNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
        self.classes = FASHION_CLASSES
        self.num_classes = NUM_CLASSES

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.inner[idx]

    @property
    def targets(self) -> list[int]:
        """Access all labels for stratification purposes."""
        return self.inner.targets.tolist() if isinstance(
            self.inner.targets, torch.Tensor
        ) else list(self.inner.targets)


def stratified_split(
    dataset: FashionDesignerDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Split a dataset into train and validation subsets preserving
    the class distribution (stratified split).

    Parameters
    ----------
    dataset : FashionDesignerDataset
        The full training dataset.
    val_fraction : float
        Fraction of data to use for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_subset, val_subset : (Subset, Subset)
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_id in range(dataset.num_classes):
        class_mask = np.where(targets == class_id)[0]
        rng.shuffle(class_mask)
        n_val = max(1, int(len(class_mask) * val_fraction))
        val_indices.extend(class_mask[:n_val].tolist())
        train_indices.extend(class_mask[n_val:].tolist())

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
