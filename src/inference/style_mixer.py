"""
Style Mixer
=============
Combines generative model output with Neural Style Transfer
to create culturally-styled fashion patterns.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from src.models.style_transfer import NeuralStyleTransfer
from src.inference.generator import FashionGenerator


class StyleMixer:
    """Generate a base fashion pattern, then apply cultural style via NST.

    Pipeline: Generator → upscale → grayscale-to-RGB → NST → output
    """

    def __init__(self, generator: FashionGenerator,
                 device: torch.device | None = None) -> None:
        self.generator = generator
        self.device = device or generator.device
        self.nst = NeuralStyleTransfer(device=self.device)

    def _to_rgb(self, tensor: torch.Tensor, size: int = 256) -> torch.Tensor:
        """Upscale grayscale generated image to RGB for NST."""
        # Normalize to [0, 1]
        img = (tensor + 1.0) / 2.0 if tensor.min() < 0 else tensor
        # Upscale
        img = torch.nn.functional.interpolate(
            img, size=(size, size), mode="bilinear", align_corners=False)
        # Grayscale → RGB
        if img.size(1) == 1:
            img = img.repeat(1, 3, 1, 1)
        return img

    def generate_styled(
        self,
        style_image_path: str | Path,
        num_samples: int = 1,
        content_weight: float = 1.0,
        style_weight: float = 1e5,
        num_steps: int = 200,
        class_label: str | int | None = None,
        output_size: int = 256,
    ) -> list[torch.Tensor]:
        """Generate and style-transfer fashion images.

        Parameters
        ----------
        style_image_path : str or Path
            Path to the style reference image.
        num_samples : int
            Number of images to generate.
        content_weight, style_weight : float
            NST loss weights.
        num_steps : int
            NST optimization steps.
        class_label : str or int, optional
            For conditional generation.
        output_size : int
            Output resolution.

        Returns
        -------
        list of styled image tensors [1, 3, H, W] in [0, 1]
        """
        # Load style image
        style_tensor = NeuralStyleTransfer.load_image(
            str(style_image_path), size=output_size)

        # Generate base patterns
        base_images = self.generator.generate(num_samples, class_label)

        styled_results = []
        for i in range(num_samples):
            content = self._to_rgb(base_images[i:i+1], size=output_size)

            def progress_cb(step, c_loss, s_loss, tv_loss):
                print(f"  NST Step {step}: content={c_loss:.4f} "
                      f"style={s_loss:.6f} tv={tv_loss:.6f}")

            result = self.nst.apply_style(
                content, style_tensor,
                content_weight=content_weight,
                style_weight=style_weight,
                num_steps=num_steps,
                progress_callback=progress_cb)

            styled_results.append(result)

        return styled_results

    @staticmethod
    def save_results(images: list[torch.Tensor],
                     output_dir: str | Path, prefix: str = "styled") -> None:
        """Save styled images to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            pil_img = NeuralStyleTransfer.tensor_to_image(img)
            pil_img.save(output_dir / f"{prefix}_{i:03d}.png")
