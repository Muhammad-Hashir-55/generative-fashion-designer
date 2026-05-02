"""
Gemini Vision Baseline
======================
Uses Google Gemini Vision API to generate texture/fashion reference images
as a comparison baseline against our trained generative models.

When a Gemini API key is available, queries gemini-2.0-flash-exp to describe
and retrieve a reference image for a given texture class.
Falls back to serving a real dataset sample when the API is unavailable.
"""

from __future__ import annotations

import base64
import io
import os
import json
import hashlib
from pathlib import Path
from typing import Optional

# Suppress protobuf version conflicts from TF/google-generativeai coexistence
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from PIL import Image


# Cache directory for Gemini responses
_CACHE_DIR = Path("./outputs/gemini_cache")


def _get_api_key() -> Optional[str]:
    """Get Gemini API key from environment variable."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()[:16]


def _load_cached(key: str) -> Optional[str]:
    """Load cached base64 image."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f).get("b64")
    return None


def _save_cached(key: str, b64: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump({"b64": b64}, f)


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str) -> Image.Image:
    """Convert base64 string to PIL image."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def generate_gemini_reference(
    class_name: str,
    image_size: int = 256,
    use_cache: bool = True,
) -> dict:
    """
    Generate a reference image using Gemini Vision API for comparison.

    Parameters
    ----------
    class_name : str
        DTD texture class name (e.g. "knitted", "striped", "woven").
    image_size : int
        Target size for the returned reference image.
    use_cache : bool
        Whether to use cached results.

    Returns
    -------
    dict with keys:
        'b64': base64-encoded PNG
        'source': 'gemini' | 'fallback'
        'description': text description
    """
    prompt_key = _cache_key(f"gemini_ref_{class_name}_{image_size}")

    if use_cache:
        cached = _load_cached(prompt_key)
        if cached:
            return {"b64": cached, "source": "gemini_cached", "description": f"Cached Gemini reference for '{class_name}' texture"}

    api_key = _get_api_key()

    if api_key:
        try:
            return _call_gemini_api(class_name, image_size, api_key, prompt_key)
        except Exception as e:
            print(f"[GeminiBaseline] API call failed: {e}. Falling back to synthetic reference.")

    # Fallback: generate a synthetic reference image using PIL
    return _generate_synthetic_reference(class_name, image_size)


def _call_gemini_api(class_name: str, image_size: int, api_key: str, cache_key_str: str) -> dict:
    """Call Gemini API to get a texture reference image."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    prompt = (
        f"Generate a high-quality, close-up photograph of a '{class_name}' texture/fabric pattern. "
        f"The image should clearly show the characteristic features of {class_name} textile texture. "
        f"Photorealistic, well-lit, centered composition, no text or watermarks."
    )

    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "image/png"}
    )

    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                b64 = base64.b64encode(part.inline_data.data).decode()
                # Resize to target size
                img = _b64_to_pil(b64).resize((image_size, image_size), Image.LANCZOS)
                b64_resized = _pil_to_b64(img)
                _save_cached(cache_key_str, b64_resized)
                return {
                    "b64": b64_resized,
                    "source": "gemini",
                    "description": f"Gemini Vision reference for '{class_name}' texture"
                }

    raise ValueError("No image in Gemini response")


def _generate_synthetic_reference(class_name: str, image_size: int) -> dict:
    """
    Generate a synthetic reference image using procedural patterns.
    Used as fallback when Gemini API is unavailable.
    """
    import numpy as np

    rng = np.random.RandomState(hash(class_name) % (2**31))

    img_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Generate texture-specific synthetic patterns
    texture_generators = {
        "striped": _gen_striped,
        "dotted": _gen_dotted,
        "chequered": _gen_chequered,
        "grid": _gen_grid,
        "woven": _gen_woven,
        "knitted": _gen_knitted,
        "zigzagged": _gen_zigzagged,
        "polka-dotted": _gen_dotted,
        "honeycombed": _gen_honeycombed,
    }

    gen_fn = texture_generators.get(class_name.lower(), _gen_default)
    img_array = gen_fn(image_size, rng)

    img = Image.fromarray(img_array)
    b64 = _pil_to_b64(img)

    return {
        "b64": b64,
        "source": "synthetic_fallback",
        "description": f"Synthetic procedural reference for '{class_name}' (Gemini API not configured)"
    }


def _gen_striped(size: int, rng) -> "np.ndarray":
    import numpy as np
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c1 = rng.randint(50, 200, 3)
    c2 = rng.randint(50, 200, 3)
    stripe_w = rng.randint(8, 24)
    for y in range(size):
        color = c1 if (y // stripe_w) % 2 == 0 else c2
        img[y, :] = color
    return img


def _gen_dotted(size: int, rng) -> "np.ndarray":
    import numpy as np
    img = np.ones((size, size, 3), dtype=np.uint8) * rng.randint(200, 240, 3).astype(np.uint8)
    dot_color = rng.randint(20, 120, 3).astype(np.uint8)
    spacing = rng.randint(12, 28)
    radius = spacing // 3
    for y in range(spacing // 2, size, spacing):
        for x in range(spacing // 2, size, spacing):
            yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = yy**2 + xx**2 <= radius**2
            y0, y1 = max(0, y-radius), min(size, y+radius+1)
            x0, x1 = max(0, x-radius), min(size, x+radius+1)
            my0, mx0 = y0-(y-radius), x0-(x-radius)
            img[y0:y1, x0:x1][mask[my0:my0+(y1-y0), mx0:mx0+(x1-x0)]] = dot_color
    return img


def _gen_chequered(size: int, rng) -> "np.ndarray":
    import numpy as np
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c1 = rng.randint(20, 80, 3).astype(np.uint8)
    c2 = rng.randint(180, 240, 3).astype(np.uint8)
    sq = rng.randint(16, 40)
    for y in range(size):
        for x in range(size):
            img[y, x] = c1 if ((y // sq) + (x // sq)) % 2 == 0 else c2
    return img


def _gen_grid(size: int, rng) -> "np.ndarray":
    import numpy as np
    bg = rng.randint(200, 240, 3).astype(np.uint8)
    img = np.ones((size, size, 3), dtype=np.uint8) * bg
    line_color = rng.randint(30, 100, 3).astype(np.uint8)
    spacing = rng.randint(12, 32)
    for i in range(0, size, spacing):
        img[i, :] = line_color
        img[:, i] = line_color
    return img


def _gen_woven(size: int, rng) -> "np.ndarray":
    import numpy as np
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c1 = rng.randint(100, 180, 3).astype(np.uint8)
    c2 = rng.randint(60, 140, 3).astype(np.uint8)
    thread = rng.randint(4, 10)
    for y in range(size):
        for x in range(size):
            phase = (x // thread + y // thread) % 2
            # simulate over/under weave
            if (y // thread) % 2 == 0:
                img[y, x] = c1 if phase == 0 else c2
            else:
                img[y, x] = c2 if phase == 0 else c1
    return img


def _gen_knitted(size: int, rng) -> "np.ndarray":
    import numpy as np
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c1 = rng.randint(80, 200, 3).astype(np.uint8)
    c2 = rng.randint(40, 160, 3).astype(np.uint8)
    for y in range(size):
        for x in range(size):
            v = np.sin(y * 0.4 + np.cos(x * 0.3) * 2) * 0.5 + 0.5
            img[y, x] = (c1 * v + c2 * (1 - v)).astype(np.uint8)
    return img


def _gen_zigzagged(size: int, rng) -> "np.ndarray":
    import numpy as np
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c1 = rng.randint(50, 200, 3).astype(np.uint8)
    c2 = rng.randint(50, 200, 3).astype(np.uint8)
    period = rng.randint(16, 40)
    for y in range(size):
        for x in range(size):
            zag = abs((x % period) - period // 2)
            img[y, x] = c1 if (y + zag) % (period // 2) < period // 4 else c2
    return img


def _gen_honeycombed(size: int, rng) -> "np.ndarray":
    import numpy as np
    img = np.zeros((size, size, 3), dtype=np.uint8)
    bg = rng.randint(200, 230, 3).astype(np.uint8)
    fc = rng.randint(100, 180, 3).astype(np.uint8)
    img[:] = bg
    r = rng.randint(12, 22)
    h = int(r * 1.732)
    w = r * 2
    for row in range(-1, size // h + 2):
        for col in range(-1, size // w + 2):
            cx = col * w + (row % 2) * r
            cy = row * h
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) * 0.577 + abs(dy) < r:
                        py, px = cy + dy, cx + dx
                        if 0 <= py < size and 0 <= px < size:
                            img[py, px] = fc
    return img


def _gen_default(size: int, rng) -> "np.ndarray":
    import numpy as np
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c1 = rng.randint(60, 200, 3).astype(np.uint8)
    c2 = rng.randint(60, 200, 3).astype(np.uint8)
    for y in range(size):
        for x in range(size):
            v = (np.sin(x * 0.2) * np.cos(y * 0.2) + 1) / 2
            img[y, x] = (c1 * v + c2 * (1 - v)).astype(np.uint8)
    return img
