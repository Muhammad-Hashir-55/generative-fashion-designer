"""
Build a text-only gallery manifest for deployment environments.

Hugging Face Spaces can reject raw image binaries in deployment pushes, so this
script converts gallery images into base64 JSON that the Flask app can serve as
an API fallback when the image files are not shipped.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path


SUPPORTED_SUFFIXES = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
}


def _gallery_model_and_timestamp_from_stem(stem: str, fallback_timestamp: int) -> tuple[str, int]:
    parts = stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1]), int(parts[-1])
    return (parts[0] if parts else "unknown"), fallback_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deploy-time gallery manifest")
    parser.add_argument("--source", type=str, default="outputs/gallery",
                        help="Directory containing gallery images")
    parser.add_argument("--output", type=str, default="app/gallery_seed_manifest.json",
                        help="Destination JSON manifest path")
    return parser.parse_args()


def _build_item(path: Path) -> dict:
    fallback_timestamp = int(path.stat().st_mtime)
    model, timestamp = _gallery_model_and_timestamp_from_stem(
        path.stem, fallback_timestamp=fallback_timestamp)
    suffix = path.suffix.lower()
    output_format = SUPPORTED_SUFFIXES[suffix]

    return {
        "id": path.stem,
        "model": model,
        "timestamp": timestamp,
        "filename": path.name,
        "format": output_format,
        "b64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def build_manifest(source_dir: Path) -> list[dict]:
    if not source_dir.exists():
        return []

    items = []
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            items.append(_build_item(path))

    items.sort(key=lambda item: item["timestamp"], reverse=True)
    return items


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": int(time.time()),
        "gallery": build_manifest(source_dir),
    }

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=True, separators=(",", ":"))

    print(f"Wrote {len(payload['gallery'])} gallery entries to {output_path}")


if __name__ == "__main__":
    main()
