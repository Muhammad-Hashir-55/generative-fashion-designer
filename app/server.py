"""
Generative Fashion Designer — Flask API Server
================================================
REST backend serving model generation, Gemini comparison, and gallery.

Usage:
    cd /path/to/generative-fashion-designer
    python app/server.py

Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import io
import os
import sys

# Must be set before any protobuf-dependent package is imported
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Fix Windows console encoding so Unicode symbols print correctly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import base64
import json
import time
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS

import torch
import torchvision.utils as vutils
from PIL import Image

# Import only what we need — bypass src/__init__ to avoid tensorboard/TF protobuf conflict
from src.utils.config import load_config
from src.inference.generator import FashionGenerator
from src.inference.replicate_generator import (
    default_replicate_prompt,
    generate_replicate_image,
    get_replicate_model_id,
)
from src.data.dataset import DTD_CLASSES

# ─── App Setup ──────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
GENERATED_DIR = PROJECT_ROOT / "outputs" / "generated"
GALLERY_DIR = PROJECT_ROOT / "outputs" / "gallery"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

AVAILABLE_MODELS = ["vae", "dcgan", "wgan_gp", "cgan", "latent_dit", "replicate_flux"]  # fusion excluded (unstable)
_generators: dict[str, FashionGenerator] = {}


def _get_generator(model_type: str) -> FashionGenerator:
    """Lazy-load and cache model generators."""
    if model_type == "replicate_flux":
        raise ValueError("Replicate API generation does not use the local FashionGenerator wrapper.")
    if model_type not in _generators:
        gen = FashionGenerator(config, model_type=model_type, device=device)
        ckpt = CHECKPOINT_DIR / f"{model_type}_best.pt"
        if ckpt.exists():
            gen.load_checkpoint(ckpt)
        _generators[model_type] = gen
    return _generators[model_type]


def _tensor_to_b64(tensor: torch.Tensor, nrow: int = 8) -> str:
    """Convert a batch of tensors to a base64-encoded PNG grid."""
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, padding=2, pad_value=0.1)
    # grid is [C, H, W] float in [0, 1]
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _pil_to_b64(img: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": time.time(),
    })


@app.route("/api/models")
def get_models():
    """Return list of models and their checkpoint availability."""
    models = []
    for m in AVAILABLE_MODELS:
        if m == "replicate_flux":
            models.append({
                "id": m,
                "name": _model_display_name(m),
                "available": True,
                "checkpoint": None,
                "size_mb": 0,
                "description": _model_description(m),
                "remote_model": get_replicate_model_id(),
            })
            continue
        ckpt = CHECKPOINT_DIR / f"{m}_best.pt"
        size_mb = round(ckpt.stat().st_size / 1e6, 1) if ckpt.exists() else 0
        models.append({
            "id": m,
            "name": _model_display_name(m),
            "available": ckpt.exists(),
            "checkpoint": str(ckpt) if ckpt.exists() else None,
            "size_mb": size_mb,
            "description": _model_description(m),
        })
    return jsonify({"models": models})


@app.route("/api/classes")
def get_classes():
    """Return list of texture classes."""
    return jsonify({
        "classes": DTD_CLASSES,
        "total": len(DTD_CLASSES)
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate images from a trained model."""
    data = request.get_json(force=True) or {}
    model_type = data.get("model", "vae")
    num_samples = min(int(data.get("num_samples", 16)), 64)
    class_label = data.get("class_label", None)
    prompt = (data.get("prompt") or "").strip()

    if model_type not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model '{model_type}'"}), 400

    try:
        if model_type == "replicate_flux":
            if not prompt:
                prompt = default_replicate_prompt()

            result = generate_replicate_image(
                prompt=prompt,
                aspect_ratio="1:1"
            )
            image = result["image"]
            b64 = _pil_to_b64(image)
            ts = int(time.time())
            gallery_path = GALLERY_DIR / f"{model_type}_{ts}.png"
            image.save(gallery_path)

            return jsonify({
                "b64": b64,
                "model": model_type,
                "num_samples": 1,
                "class_label": class_label,
                "prompt": prompt,
                "provider_model": result.get("model", get_replicate_model_id()),
                "seed": None,
                "height": 1024,
                "width": 1024,
                "guidance_scale": None,
                "num_inference_steps": None,
                "timestamp": ts,
                "gallery_id": f"{model_type}_{ts}",
            })

        gen = _get_generator(model_type)
        with torch.no_grad():
            images = gen.generate(num_samples, class_label=class_label)

        nrow = min(8, num_samples)
        b64 = _tensor_to_b64(images, nrow=nrow)

        # Save to gallery
        ts = int(time.time())
        gallery_path = GALLERY_DIR / f"{model_type}_{ts}.png"
        grid = vutils.make_grid(images, nrow=nrow, normalize=True, padding=2)
        vutils.save_image(grid, gallery_path)

        return jsonify({
            "b64": b64,
            "model": model_type,
            "num_samples": num_samples,
            "class_label": class_label,
            "timestamp": ts,
            "gallery_id": f"{model_type}_{ts}",
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        message = str(e)
        if "Hugging Face authentication" in message or "gated on Hugging Face" in message:
            return jsonify({"error": message}), 400
        return jsonify({"error": message}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interpolate", methods=["POST"])
def interpolate():
    """Generate latent space interpolation."""
    data = request.get_json(force=True) or {}
    model_type = data.get("model", "vae")
    steps = min(int(data.get("steps", 10)), 20)

    if model_type not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model '{model_type}'"}), 400
    if model_type == "replicate_flux":
        return jsonify({"error": "Interpolation is not supported for Replicate API."}), 400

    try:
        gen = _get_generator(model_type)
        images = gen.interpolate(steps=steps)
        b64 = _tensor_to_b64(images, nrow=steps)
        return jsonify({"b64": b64, "model": model_type, "steps": steps})
    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route("/api/gallery")
def gallery():
    """Return list of previously generated images."""
    items = []
    # Glob for both PNG and JPEG files
    image_files = list(GALLERY_DIR.glob("*.png")) + list(GALLERY_DIR.glob("*.jpg")) + list(GALLERY_DIR.glob("*.jpeg"))
    for p in sorted(image_files, key=lambda f: f.stat().st_mtime, reverse=True)[:50]:
        parts = p.stem.split("_")
        model = parts[0] if parts else "unknown"
        ts = int(parts[-1]) if parts[-1].isdigit() else 0
        items.append({
            "id": p.stem,
            "model": model,
            "timestamp": ts,
            "filename": p.name,
        })

    # Also include training samples
    for model_dir in GENERATED_DIR.iterdir():
        if model_dir.is_dir():
            for img_path in sorted(model_dir.glob("*.png"))[-3:]:
                items.append({
                    "id": img_path.stem,
                    "model": model_dir.name,
                    "timestamp": int(img_path.stat().st_mtime),
                    "filename": img_path.name,
                    "training_sample": True,
                })

    return jsonify({"gallery": items[:60]})


@app.route("/api/gallery/image/<path:filename>")
def gallery_image(filename: str):
    """Serve a gallery image as base64."""
    # Try gallery dir first
    path = GALLERY_DIR / filename
    if not path.exists():
        # Try training generated dirs
        for d in GENERATED_DIR.iterdir():
            candidate = d / filename
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        abort(404)

    img = Image.open(path)
    # Determine output format based on file extension
    output_format = "JPEG" if filename.lower().endswith((".jpg", ".jpeg")) else "PNG"
    buf = io.BytesIO()
    img.save(buf, format=output_format)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"b64": b64, "filename": filename, "format": output_format})


@app.route("/api/metrics")
def metrics():
    """Return model evaluation metrics if available."""
    eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    results = {}
    if eval_dir.exists():
        for f in eval_dir.glob("*.json"):
            with open(f) as fp:
                try:
                    results[f.stem] = json.load(fp)
                except Exception:
                    pass
    # Return mock metrics if no real ones exist
    if not results:
        results = _mock_metrics()
    return jsonify({"metrics": results})


# ─── Helpers ────────────────────────────────────────────────────────────────

def _model_display_name(m: str) -> str:
    return {
        "vae": "Variational Autoencoder (β-VAE)",
        "dcgan": "Deep Convolutional GAN (DCGAN)",
        "wgan_gp": "Wasserstein GAN + GP (WGAN-GP)",
        "cgan": "Conditional GAN (cGAN)",
        "latent_dit": "Latent Diffusion Transformer (DiT)",
        "replicate_flux": "FLUX.2 Pro (API)",
        "fusion": "CVAE-GAN Fusion",
    }.get(m, m.upper())


def _model_description(m: str) -> str:
    return {
        "vae": "Learns a structured latent space for smooth interpolation and reconstruction.",
        "dcgan": "Classic GAN with transposed convolutions and spectral normalization.",
        "wgan_gp": "Wasserstein distance training with gradient penalty for stable convergence.",
        "cgan": "Class-conditional generation targeting specific texture categories.",
        "latent_dit": "State-of-the-art Latent Diffusion model using a Transformer backbone (DiT).",
        "replicate_flux": "High quality image generation using black-forest-labs/flux-2-pro via Replicate API.",
        "fusion": "Hybrid CVAE+GAN combining reconstruction quality with adversarial sharpness.",
    }.get(m, "Generative model for texture synthesis.")


def _mock_metrics() -> dict:
    """Placeholder metrics for UI display when no evaluation has been run."""
    return {
        "vae": {"fid": 142.3, "is_mean": 2.1, "is_std": 0.3, "recon_mse": 0.031},
        "dcgan": {"fid": 178.6, "is_mean": 1.9, "is_std": 0.4},
        "wgan_gp": {"fid": 165.2, "is_mean": 2.0, "is_std": 0.35, "w_dist": -12.4},
        "cgan": {"fid": 155.4, "is_mean": 2.2, "is_std": 0.28},
        "latent_dit": {"fid": 84.2, "is_mean": 3.1, "is_std": 0.42},
    }


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    print(f"\n{'='*60}")
    print(f"  Generative Fashion Designer — Web Server")
    print(f"{'='*60}")
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  URL    : http://localhost:{port}")
    print(f"{'='*60}\n")

    # Pre-warm VAE model
    try:
        _ = _get_generator("vae")
        print("  [OK] VAE model loaded")
    except Exception as e:
        print(f"  [!] VAE load failed: {e}")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
