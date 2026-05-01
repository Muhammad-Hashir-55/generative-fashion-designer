"""
Dataset Module
===============
Provides access to the Describable Textures Dataset (DTD) which
serves as our proxy for high-frequency textile and fabric patterns.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.datasets import DTD


# Canonical DTD class order (47 categories). Kept as a constant so that
# inference utilities can map names -> labels without requiring dataset
# download/availability.
DTD_CLASSES: list[str] = [
    "banded",
    "blotchy",
    "braided",
    "bubbly",
    "bumpy",
    "chequered",
    "cobwebbed",
    "cracked",
    "crosshatched",
    "crystalline",
    "dotted",
    "fibrous",
    "flecked",
    "freckled",
    "frilly",
    "gauzy",
    "grid",
    "grooved",
    "honeycombed",
    "interlaced",
    "knitted",
    "lacelike",
    "lined",
    "marbled",
    "matted",
    "meshed",
    "paisley",
    "perforated",
    "pitted",
    "pleated",
    "polka-dotted",
    "porous",
    "potholed",
    "scaly",
    "smeared",
    "spiralled",
    "sprinkled",
    "stained",
    "stratified",
    "striped",
    "studded",
    "swirly",
    "veined",
    "waffled",
    "woven",
    "wrinkled",
    "zigzagged",
]

# Backwards-compatible alias used by inference code.
FASHION_CLASSES = DTD_CLASSES


class FashionDesignerDataset(Dataset):
    """Wrapper for the DTD dataset with split support."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        download: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        root : str
            Data directory.
        split : str
            One of 'train', 'val', 'test'.
        transform : callable, optional
            A function/transform to apply to the images.
        download : bool
            If true, downloads the dataset from the internet.
        """
        super().__init__()
        self.dataset = DTD(
            root=root,
            split=split,
            transform=transform,
            download=download,
        )
        self.classes = self.dataset.classes

        # Extract targets (labels) without iterating the dataset (which would
        # load/transform every image and is very slow).
        labels = None
        for attr in ("_labels", "labels", "targets"):
            if hasattr(self.dataset, attr):
                labels = getattr(self.dataset, attr)
                break
        self.targets = list(labels) if labels is not None else []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        return image, label


def get_class_names(root: str = "./data", prefer_dataset: bool = False) -> list[str]:
    """Return the list of 47 texture classes in DTD.

    By default this is offline-safe and returns the canonical constant list.
    Set ``prefer_dataset=True`` to query torchvision's DTD metadata if the
    dataset is already present locally.
    """
    if prefer_dataset:
        try:
            ds = DTD(root=root, split="train", download=False)
            return list(ds.classes)
        except Exception:
            pass
    return list(DTD_CLASSES)


def name_to_label(name: str, root: str = "./data") -> int:
    """Convert class name to integer label."""
    classes = get_class_names(root)
    name = name.lower()
    for i, c in enumerate(classes):
        if c.lower() == name:
            return i
    raise ValueError(f"Class '{name}' not found. Available: {classes}")
