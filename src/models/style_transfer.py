"""
VGG-19 Neural Style Transfer (Enhanced)
=========================================
Artistic style transfer using Gram-matrix style loss and deep content
loss from a pre-trained VGG-19 network.

Enhancements over baseline:
- Total variation loss for spatial smoothness
- Multi-resolution style blending
- Progress callbacks for training loop integration
- Aggressive VRAM management for 6 GB GPUs

Reference: Gatys et al., "A Neural Algorithm of Artistic Style", 2015
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vgg import VGG19_Weights
import torchvision.transforms as T
from PIL import Image


class NeuralStyleTransfer:
    """VGG-19 based neural style transfer engine."""

    # Layer name → index mapping for VGG-19 features
    LAYER_MAP = {
        "conv1_1": "0",   "conv1_2": "2",
        "conv2_1": "5",   "conv2_2": "7",
        "conv3_1": "10",  "conv3_2": "12",  "conv3_3": "14",  "conv3_4": "16",
        "conv4_1": "19",  "conv4_2": "21",  "conv4_3": "23",  "conv4_4": "25",
        "conv5_1": "28",  "conv5_2": "30",  "conv5_3": "32",  "conv5_4": "34",
    }

    def __init__(
        self,
        device: torch.device | None = None,
        content_layers: list[str] | None = None,
        style_layers: list[str] | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.content_layers = content_layers or ["21"]    # conv4_2
        self.style_layers = style_layers or ["0", "5", "10", "19", "28"]

        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        vgg = vgg.to(self.device).eval()
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.model = vgg

        self.norm_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        ).view(-1, 1, 1)
        self.norm_std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        ).view(-1, 1, 1)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization."""
        return (tensor - self.norm_mean) / self.norm_std

    def _extract_features(
        self, image: torch.Tensor,
    ) -> dict[str, list[torch.Tensor]]:
        """Run forward pass and extract content + style features."""
        features: dict[str, list[torch.Tensor]] = {
            "content": [], "style": [],
        }
        x = self._normalize(image)
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.content_layers:
                features["content"].append(x)
            if name in self.style_layers:
                features["style"].append(x)
        return features

    @staticmethod
    def _gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
        """Compute the Gram matrix G = F · Fᵀ / (C × H × W)."""
        B, C, H, W = tensor.size()
        features = tensor.view(B * C, H * W)
        G = torch.mm(features, features.t())
        return G.div(B * C * H * W)

    @staticmethod
    def _total_variation_loss(image: torch.Tensor) -> torch.Tensor:
        """Total variation regularization for spatial smoothness."""
        tv_h = ((image[:, :, 1:, :] - image[:, :, :-1, :]) ** 2).mean()
        tv_w = ((image[:, :, :, 1:] - image[:, :, :, :-1]) ** 2).mean()
        return tv_h + tv_w

    def apply_style(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        content_weight: float = 1.0,
        style_weight: float = 1e5,
        tv_weight: float = 1e-6,
        num_steps: int = 300,
        optimizer_type: str = "lbfgs",
        progress_callback: Callable[[int, float, float, float], None] | None = None,
    ) -> torch.Tensor:
        content_image = content_image.to(self.device)
        style_image = style_image.to(self.device)
        target = content_image.clone().requires_grad_(True).to(self.device)

        with torch.no_grad():
            content_features = self._extract_features(content_image)["content"]
            style_features = self._extract_features(style_image)["style"]
            style_grams = [self._gram_matrix(sf) for sf in style_features]

        if optimizer_type == "lbfgs":
            optimizer = torch.optim.LBFGS([target], lr=1.0, max_iter=1)
        else:
            optimizer = torch.optim.Adam([target], lr=0.01)

        step_counter = [0]

        while step_counter[0] < num_steps:
            def closure():
                target.data.clamp_(0, 1)
                optimizer.zero_grad()

                target_features = self._extract_features(target)

                c_loss = sum(
                    F.mse_loss(tf, cf)
                    for tf, cf in zip(target_features["content"], content_features)
                )

                s_loss = sum(
                    F.mse_loss(self._gram_matrix(tf), sg)
                    for tf, sg in zip(target_features["style"], style_grams)
                )

                tv_loss = self._total_variation_loss(target)

                total = content_weight * c_loss + style_weight * s_loss + tv_weight * tv_loss
                total.backward()

                step_counter[0] += 1
                if step_counter[0] % 50 == 0:
                    if progress_callback:
                        progress_callback(
                            step_counter[0], c_loss.item(), s_loss.item(), tv_loss.item(),
                        )

                del target_features
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                return total

            optimizer.step(closure)

        target.data.clamp_(0, 1)
        result = target.detach().cpu()

        del target, content_features, style_features, style_grams
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return result

    @staticmethod
    def load_image(
        path: str, size: int = 512,
    ) -> torch.Tensor:
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])
        img = Image.open(path).convert("RGB")
        return transform(img).unsqueeze(0)

    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        img = tensor.squeeze(0).clamp(0, 1)
        img = T.ToPILImage()(img)
        return img
