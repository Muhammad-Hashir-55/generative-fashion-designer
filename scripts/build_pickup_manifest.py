"""
Build a text-only pickup-image manifest for deployment environments.

The deployed Space can reconstruct preview generations from this manifest even
when the raw ``outputs/pickup-pictures`` JPG files are removed before the clean
push to Hugging Face.
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


def _class_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deploy-time pickup preview manifest")
    parser.add_argument("--source", type=str, default="outputs/pickup-pictures",
                        help="Directory containing curated pickup preview images")
    parser.add_argument("--output", type=str, default="app/pickup_seed_manifest.json",
                        help="Destination JSON manifest path")
    return parser.parse_args()


def build_manifest(source_dir: Path) -> list[dict]:
    if not source_dir.exists():
        return []

    items = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        items.append({
            "filename": path.name,
            "class_name": _class_name_from_filename(path.name),
            "format": SUPPORTED_SUFFIXES[path.suffix.lower()],
            "b64": base64.b64encode(path.read_bytes()).decode("ascii"),
        })
    return items


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": int(time.time()),
        "images": build_manifest(source_dir),
    }

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=True, separators=(",", ":"))

    print(f"Wrote {len(payload['images'])} pickup preview entries to {output_path}")


if __name__ == "__main__":
    main()
