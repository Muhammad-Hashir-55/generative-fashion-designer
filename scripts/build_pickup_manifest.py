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
DEFAULT_MAX_BYTES = 9 * 1024 * 1024


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
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                        help="Maximum size for a single manifest file")
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


def _manifest_family_paths(output_path: Path) -> list[Path]:
    pattern = f"{output_path.stem}*.json"
    return list(output_path.parent.glob(pattern))


def _payload_bytes(images: list[dict], generated_at: int,
                   chunk_index: int | None = None,
                   chunk_count: int | None = None) -> bytes:
    payload = {
        "generated_at": generated_at,
        "images": images,
    }
    if chunk_index is not None and chunk_count is not None:
        payload["chunk_index"] = chunk_index
        payload["chunk_count"] = chunk_count
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _chunk_output_path(output_path: Path, chunk_index: int) -> Path:
    return output_path.with_name(f"{output_path.stem}.{chunk_index:03d}{output_path.suffix}")


def write_manifest_files(items: list[dict], output_path: Path, max_bytes: int) -> list[Path]:
    generated_at = int(time.time())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for old_path in _manifest_family_paths(output_path):
        old_path.unlink()

    single_payload = _payload_bytes(items, generated_at)
    if len(single_payload) <= max_bytes:
        output_path.write_bytes(single_payload)
        return [output_path]

    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []

    for item in items:
        candidate = current_chunk + [item]
        if current_chunk and len(_payload_bytes(candidate, generated_at)) > max_bytes:
            chunks.append(current_chunk)
            current_chunk = [item]
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)

    written_paths: list[Path] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_path = _chunk_output_path(output_path, idx)
        chunk_payload = _payload_bytes(chunk, generated_at, idx, len(chunks))
        if len(chunk_payload) > max_bytes:
            raise ValueError(
                f"Chunk {idx} still exceeds max size ({len(chunk_payload)} bytes > {max_bytes})."
            )
        chunk_path.write_bytes(chunk_payload)
        written_paths.append(chunk_path)

    return written_paths


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_path = Path(args.output)
    items = build_manifest(source_dir)
    written_paths = write_manifest_files(items, output_path, args.max_bytes)
    paths_str = ", ".join(path.name for path in written_paths)
    print(f"Wrote {len(items)} pickup preview entries to {paths_str}")


if __name__ == "__main__":
    main()
