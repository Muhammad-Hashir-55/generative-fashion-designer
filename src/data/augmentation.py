"""
Data Augmentation Pipelines
=============================
Defines augmentation strategies for training, evaluation, and GAN-specific
data preparation adapted for RGB texture images (DTD).

Augmentations target 3-channel RGB images and default to 64×64 resolution.
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image


class TrainAugmentation:
    """Heavy augmentation pipeline for classifier/VAE training.

    Applies spatial jittering, random erasing, and intensity perturbations
    to regularize the model during training.
    """

    def __init__(self, image_size: int = 64) -> None:
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            # Image-level adjustments operate on PIL images
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
            T.ToTensor(),
            T.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            # Use ImageNet normalization for pre-trained backbones and stability
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class EvalAugmentation:
    """Minimal transform for validation / test evaluation."""

    def __init__(self, image_size: int = 64) -> None:
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class GANAugmentation:
    """GAN-specific augmentation that normalizes to [−1, 1].

    Uses Tanh-compatible range so Generator output with Tanh activation
    directly matches the data distribution.
    """

    def __init__(self, image_size: int = 64) -> None:
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            # Normalize to [-1, 1] to match common GAN generator outputs
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),   # → [-1, 1]
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class StyleTransferAugmentation:
    """Transforms for neural style transfer (3-channel, ImageNet normalised).

    Converts grayscale → RGB by repeating channels, then applies ImageNet
    normalization expected by VGG-19.
    """

    def __init__(self, image_size: int = 256) -> None:
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            # VGG-19 ImageNet normalization
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)
