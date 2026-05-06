"""
Replicate API text-to-image integration.

This module uses the Replicate API to run black-forest-labs/flux-2-pro.
"""

from __future__ import annotations

import os
import io
import replicate
from PIL import Image
from dotenv import load_dotenv

_MODEL_ID = "black-forest-labs/flux-2-pro"

# Pre-load environment variables
load_dotenv()
def get_replicate_model_id() -> str:
    return _MODEL_ID

def default_replicate_prompt() -> str:
    return (
        "Create a premium square textile swatch inspired by Pakistani Ajrak motifs, "
        "deep indigo and madder red, intricate geometric block-print pattern, rich fabric detail, "
        "studio lighting, centered composition, no text, no watermark."
    )

def generate_replicate_image(
    prompt: str,
    *,
    aspect_ratio: str = "1:1",
    output_quality: int = 100,
    safety_tolerance: int = 2,
) -> dict[str, object]:
    """Generate a single image via Replicate API."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("A prompt is required for Replicate generation.")

    # Ensure the API token is loaded via environment variables
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN environment variable not set. Please add it to your .env file.")

    try:
        output = replicate.run(
            _MODEL_ID,
            input={
                "prompt": prompt,
                "resolution": "1 MP",
                "aspect_ratio": aspect_ratio,
                "input_images": [],
                "output_format": "png",
                "output_quality": output_quality,
                "safety_tolerance": safety_tolerance
            }
        )
        
        # Output contains the image bytes from output.read()
        image_bytes = output.read()
        image = Image.open(io.BytesIO(image_bytes))

        return {
            "image": image,
            "prompt": prompt,
            "model": _MODEL_ID,
            "aspect_ratio": aspect_ratio,
            "output_quality": output_quality,
        }
    except Exception as exc:
        raise RuntimeError(f"Replicate API generation failed: {exc}") from exc
