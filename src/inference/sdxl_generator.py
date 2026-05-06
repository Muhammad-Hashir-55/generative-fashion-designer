"""
SDXL Turbo text-to-image integration.

This module uses the stabilityai/sdxl-turbo model for extremely fast, low-VRAM image generation.
"""

from __future__ import annotations

import threading
import torch
from diffusers import AutoPipelineForText2Image

_MODEL_ID = "stabilityai/sdxl-turbo"
_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()

def get_sdxl_model_id() -> str:
    return _MODEL_ID

def default_sdxl_prompt() -> str:
    return (
        "Create a premium square textile swatch inspired by Pakistani Ajrak motifs, "
        "deep indigo and madder red, intricate geometric block-print pattern, rich fabric detail, "
        "studio lighting, centered composition, no text, no watermark."
    )

def _build_pipeline(device: torch.device):
    kwargs = {
        "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
    }

    pipe = AutoPipelineForText2Image.from_pretrained(_MODEL_ID, **kwargs)
    pipe = pipe.to(device)

    return pipe

def get_pipeline(device: torch.device | None = None):
    global _PIPELINE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = _build_pipeline(device)
    return _PIPELINE

def generate_sdxl_image(
    prompt: str,
    *,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 0.0,
    num_inference_steps: int = 1,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Generate a single image via SDXL Turbo."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("A prompt is required for SDXL Turbo generation.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        pipe = get_pipeline(device)
        generator = torch.Generator(device.type).manual_seed(int(seed))

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                generator=generator,
            )
        image = result.images[0]
        return {
            "image": image,
            "prompt": prompt,
            "model": _MODEL_ID,
            "seed": int(seed),
            "height": height,
            "width": width,
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
        }
    except Exception as exc:
        raise RuntimeError(f"SDXL Turbo generation failed: {exc}") from exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
