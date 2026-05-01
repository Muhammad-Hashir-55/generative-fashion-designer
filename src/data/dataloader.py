"""
DataLoader Factory
===================
One-call factory that returns ready-to-use train/val/test DataLoaders
for any model type, applying the correct augmentation pipeline.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Subset

from src.data.dataset import FashionDesignerDataset, stratified_split
from src.data.augmentation import (
    TrainAugmentation,
    EvalAugmentation,
    GANAugmentation,
)
from src.utils.config import Config


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
    val_split = getattr(data_cfg, "val_split", 0.1)
    data_dir = getattr(data_cfg, "data_dir", "./data")
    seed = getattr(config.project, "seed", 42)

    # ── Select augmentation pipeline ──────────────────────────────────────
    if mode == "gan":
        train_transform = GANAugmentation(image_size)
    elif mode == "train":
        train_transform = TrainAugmentation(image_size)
    else:
        train_transform = EvalAugmentation(image_size)

    eval_transform = EvalAugmentation(image_size)

    # ── Build datasets ────────────────────────────────────────────────────
    full_train = FashionDesignerDataset(
        root=data_dir, train=True, transform=train_transform, download=True,
    )
    train_subset, val_subset_raw = stratified_split(
        full_train, val_fraction=val_split, seed=seed,
    )

    # Validation uses eval transform — wrap with a transform-swapping subset
    full_train_eval = FashionDesignerDataset(
        root=data_dir, train=True, transform=eval_transform, download=True,
    )
    val_subset = Subset(full_train_eval, val_subset_raw.indices)

    test_dataset = FashionDesignerDataset(
        root=data_dir, train=False, transform=eval_transform, download=True,
    )

    # ── Build DataLoaders ─────────────────────────────────────────────────
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    loaders = {
        "train": DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    }

    return loaders
