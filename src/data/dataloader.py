"""
DataLoader Factory
===================
One-call factory that returns ready-to-use train/val/test DataLoaders
for any model type, applying the correct augmentation pipeline.
"""

from __future__ import annotations

from functools import partial

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import FashionDesignerDataset
from src.data.augmentation import (
    TrainAugmentation,
    EvalAugmentation,
    GANAugmentation,
)
from src.utils.config import Config


def seed_worker(worker_id: int, seed: int) -> None:
    """Seed dataloader workers in a Windows-pickle-friendly way."""
    import random

    import numpy as np
    import torch

    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloaders(
    config: Config,
    mode: str = "gan",
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders.

    Parameters
    ----------
    config : Config
        Project configuration (needs ``config.data.*``).
    mode : str
        One of ``"train"`` (heavy augmentation), ``"gan"`` (GAN-specific),
        or ``"eval"`` (minimal). Controls the augmentation pipeline.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    data_cfg = config.data
    image_size = getattr(data_cfg, "image_size", 32)
    batch_size = getattr(data_cfg, "batch_size", 128)
    num_workers = getattr(data_cfg, "num_workers", 4)
    pin_memory = getattr(data_cfg, "pin_memory", True)
    # NOTE: DTD provides predefined train/val/test splits.
    data_dir = getattr(data_cfg, "data_dir", "./data")
    seed = getattr(config.project, "seed", 42)

    persistent_workers = getattr(
        data_cfg,
        "persistent_workers",
        bool(num_workers and num_workers > 0),
    )
    prefetch_factor = getattr(data_cfg, "prefetch_factor", 2)

    # ── Select augmentation pipeline ──────────────────────────────────────
    if mode == "gan":
        train_transform = GANAugmentation(image_size)
    elif mode == "train":
        train_transform = TrainAugmentation(image_size)
    else:
        train_transform = EvalAugmentation(image_size)

    # For GAN mode we must use the GAN augmentation for validation as well
    # so that images are normalized to [-1, 1] and loss targets remain valid.
    if mode == "gan":
        eval_transform = GANAugmentation(image_size)
    else:
        eval_transform = EvalAugmentation(image_size)

    # ── Build datasets ────────────────────────────────────────────────────
    train_subset = FashionDesignerDataset(
        root=data_dir, split="train", transform=train_transform, download=True,
    )
    val_subset = FashionDesignerDataset(
        root=data_dir, split="val", transform=eval_transform, download=True,
    )
    test_dataset = FashionDesignerDataset(
        root=data_dir, split="test", transform=eval_transform, download=True,
    )

    # ── Build DataLoaders ─────────────────────────────────────────────────
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=partial(seed_worker, seed=seed) if num_workers and num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers and num_workers > 0 else False,
    )
    if num_workers and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loaders = {
        "train": DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        ),
    }

    return loaders
