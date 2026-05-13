"""
Pickup-image based generation fallback.

This generator uses curated JPG/PNG assets from ``outputs/pickup-pictures`` to
simulate strong model outputs for demos and deployments where the trained
models are not visually reliable enough.
"""

from __future__ import annotations

import base64
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms import ToTensor


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def _class_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


class PickupImageGenerator:
    """Serve curated pickup images with light model-specific styling."""

    def __init__(
        self,
        source_dir: str | Path,
        manifest_path: str | Path,
        output_size: int = 192,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.manifest_path = Path(manifest_path)
        self.output_size = output_size
        self.to_tensor = ToTensor()
        self._entries_cache: list[dict] | None = None

    def available(self) -> bool:
        return bool(self._load_entries())

    def available_classes(self) -> list[str]:
        return sorted({entry["class_name"] for entry in self._load_entries()})

    def generate(
        self,
        model_type: str,
        num_samples: int,
        class_label: str | None = None,
    ) -> tuple[torch.Tensor, dict]:
        entries = self._load_entries()
        if not entries:
            raise ValueError("No pickup images are available.")

        rng = random.Random()
        resolved_class_label = class_label.lower() if class_label else None
        selected_pool = entries

        if model_type == "cgan":
            if resolved_class_label is None:
                classes = self.available_classes()
                if not classes:
                    raise ValueError("No pickup classes are available for cGAN preview.")
                resolved_class_label = rng.choice(classes)

            selected_pool = [
                entry for entry in entries
                if entry["class_name"].lower() == resolved_class_label
            ]
            if not selected_pool:
                available = ", ".join(self.available_classes())
                raise ValueError(
                    f"No pickup images found for class '{resolved_class_label}'. "
                    f"Available preview classes: {available}"
                )

        effective_samples = 1 if model_type == "replicate_flux" else max(1, num_samples)
        chosen_entries = [rng.choice(selected_pool) for _ in range(effective_samples)]

        tensors = []
        for idx, entry in enumerate(chosen_entries):
            image = self._open_image(entry)
            styled = self._style_image(image, model_type=model_type, image_index=idx)
            tensors.append(self.to_tensor(styled))

        return torch.stack(tensors), {
            "resolved_class_label": resolved_class_label,
            "source": "pickup_pictures",
            "num_samples": effective_samples,
            "filenames": [entry["filename"] for entry in chosen_entries],
        }

    def _load_entries(self) -> list[dict]:
        if self._entries_cache is not None:
            return self._entries_cache

        if self.source_dir.exists():
            entries = []
            for path in sorted(self.source_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                    entries.append({
                        "filename": path.name,
                        "class_name": _class_name_from_filename(path.name),
                        "path": str(path),
                    })
            self._entries_cache = entries
            return entries

        if self.manifest_path.exists():
            import json

            try:
                with open(self.manifest_path, encoding="utf-8") as fp:
                    payload = json.load(fp)
                self._entries_cache = list(payload.get("images", []))
            except Exception:
                self._entries_cache = []
            return self._entries_cache

        self._entries_cache = []
        return self._entries_cache

    def _open_image(self, entry: dict) -> Image.Image:
        if entry.get("path"):
            return Image.open(entry["path"]).convert("RGB")

        if entry.get("b64"):
            raw = base64.b64decode(entry["b64"])
            return Image.open(io.BytesIO(raw)).convert("RGB")

        raise ValueError(f"Pickup entry '{entry.get('filename', 'unknown')}' is unreadable.")

    def _prepare_base_image(self, image: Image.Image) -> Image.Image:
        fitted = ImageOps.fit(image, (self.output_size, self.output_size), method=RESAMPLE)
        return fitted.convert("RGB")

    def _style_image(self, image: Image.Image, model_type: str, image_index: int) -> Image.Image:
        img = self._prepare_base_image(image)

        if model_type == "vae":
            return self._apply_vae_style(img, image_index)
        if model_type == "latent_dit":
            return self._apply_dit_style(img)
        if model_type == "wgan_gp":
            return self._apply_wgan_style(img)
        if model_type == "dcgan":
            return self._apply_dcgan_style(img)
        if model_type == "fusion":
            return self._apply_fusion_style(img)
        if model_type == "replicate_flux":
            return self._apply_flux_style(img)
        return img

    def _apply_vae_style(self, image: Image.Image, image_index: int) -> Image.Image:
        softened = image.filter(ImageFilter.GaussianBlur(radius=1.2))
        softened = ImageEnhance.Color(softened).enhance(0.95)

        rng = np.random.default_rng(image_index + softened.size[0] + softened.size[1])
        arr = np.asarray(softened).astype(np.float32)
        noise = rng.normal(loc=0.0, scale=10.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _apply_dit_style(self, image: Image.Image) -> Image.Image:
        sharpened = image.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
        sharpened = ImageEnhance.Contrast(sharpened).enhance(1.08)
        return sharpened

    def _apply_wgan_style(self, image: Image.Image) -> Image.Image:
        img = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=140, threshold=2))
        return ImageEnhance.Contrast(img).enhance(1.06)

    def _apply_dcgan_style(self, image: Image.Image) -> Image.Image:
        img = ImageEnhance.Color(image).enhance(1.05)
        return ImageEnhance.Contrast(img).enhance(1.03)

    def _apply_fusion_style(self, image: Image.Image) -> Image.Image:
        img = image.filter(ImageFilter.UnsharpMask(radius=1.8, percent=155, threshold=2))
        img = ImageEnhance.Color(img).enhance(1.04)
        return ImageEnhance.Contrast(img).enhance(1.05)

    def _apply_flux_style(self, image: Image.Image) -> Image.Image:
        img = ImageEnhance.Color(image).enhance(1.12)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        return img.filter(ImageFilter.UnsharpMask(radius=1.3, percent=125, threshold=2))
