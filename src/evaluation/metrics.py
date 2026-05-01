"""
Evaluation Metrics
===================
Manual implementations of FID, Inception Score, and SSIM that work
without external dependencies like torchmetrics or torch-fidelity.
Uses InceptionV3 features from torchvision.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionFeatureExtractor(nn.Module):
    """Extract 2048-dim feature vectors from InceptionV3 pool layer."""

    def __init__(self, device: torch.device | None = None) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.eval()
        # Remove final FC layer
        model.fc = nn.Identity()
        self.model = model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features. Images should be [B, 3, 299, 299] in [0, 1]."""
        x = images.to(self.device)
        # Resize if needed
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # Repeat channels if grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.model(x)


class FIDScore:
    """Fréchet Inception Distance between real and generated distributions.
    
    FID = ‖μ_r - μ_g‖² + Tr(Σ_r + Σ_g - 2(Σ_r · Σ_g)^(1/2))
    Lower is better.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.extractor = InceptionFeatureExtractor(device)

    @torch.no_grad()
    def compute(self, real_images: torch.Tensor,
                fake_images: torch.Tensor) -> float:
        """Compute FID between two batches of images."""
        feat_real = self.extractor(real_images).cpu().numpy()
        feat_fake = self.extractor(fake_images).cpu().numpy()

        mu_r, sigma_r = feat_real.mean(axis=0), np.cov(feat_real, rowvar=False)
        mu_g, sigma_g = feat_fake.mean(axis=0), np.cov(feat_fake, rowvar=False)

        diff = mu_r - mu_g
        # Stable matrix sqrt via eigendecomposition
        product = sigma_r @ sigma_g
        eigvals, eigvecs = np.linalg.eigh(product)
        eigvals = np.maximum(eigvals, 0)  # numerical stability
        sqrt_product = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * sqrt_product)
        return float(np.real(fid))


class InceptionScore:
    """Inception Score: exp(E[KL(p(y|x) ‖ p(y))])
    
    Higher is better. Measures quality and diversity.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.eval()
        self.model = model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute(self, images: torch.Tensor, splits: int = 10) -> tuple[float, float]:
        """Compute IS (mean ± std over splits)."""
        x = images.to(self.device)
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        preds = F.softmax(self.model(x), dim=1).cpu().numpy()

        scores = []
        N = len(preds)
        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits)]
            if len(part) == 0:
                continue
            p_y = np.mean(part, axis=0, keepdims=True)
            kl = part * (np.log(part + 1e-16) - np.log(p_y + 1e-16))
            scores.append(np.exp(kl.sum(axis=1).mean()))

        return float(np.mean(scores)), float(np.std(scores))


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor,
                 window_size: int = 11) -> float:
    """Compute mean SSIM between two image batches.
    
    Simplified implementation using Gaussian window.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)

    channels = img1.size(1)
    window = window.repeat(channels, 1, 1, 1).to(img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean().item())
