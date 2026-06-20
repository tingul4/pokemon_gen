from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm

from src.data.caption_builder import build_caption
from src.utils.config import PROJECT_ROOT, ensure_dir


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def find_images(raw_dir: str | Path) -> list[Path]:
    root = Path(raw_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def load_metadata(path: str | Path = "data/processed/metadata.json") -> dict[str, dict[str, Any]]:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        metadata[_slug(record["name"])] = record
        metadata[str(record["id"])] = record
    return metadata


def _match_metadata(image_path: Path, metadata: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    stem = _slug(image_path.stem)
    numeric = re.match(r"^(\d+)", stem)
    if numeric and numeric.group(1) in metadata:
        return metadata[numeric.group(1)]
    for key, record in metadata.items():
        if not key.isdigit() and key and (key == stem or key in stem):
            return record
    return None


def prepare_lora_dataset(
    raw_image_dir: str | Path = "data/raw/kaggle_pokemon_images",
    output_image_dir: str | Path = "data/processed/lora_images",
    captions_path: str | Path = "data/processed/captions.jsonl",
    metadata_path: str | Path = "data/processed/metadata.json",
    max_images: int | None = None,
    resolution: int = 768,
) -> list[dict[str, str]]:
    images = find_images(raw_image_dir)
    if max_images:
        images = images[:max_images]
    metadata = load_metadata(metadata_path)
    output_dir = ensure_dir(output_image_dir)
    captions_target = Path(captions_path)
    if not captions_target.is_absolute():
        captions_target = PROJECT_ROOT / captions_target
    captions_target.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for idx, image_path in enumerate(tqdm(images, desc="Preparing LoRA images")):
        record = _match_metadata(image_path, metadata)
        out_name = f"{idx:05d}{image_path.suffix.lower() if image_path.suffix else '.png'}"
        out_path = output_dir / out_name
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (resolution, resolution), (255, 255, 255))
                left = (resolution - img.width) // 2
                top = (resolution - img.height) // 2
                canvas.paste(img, (left, top))
                canvas.save(out_path)
        except Exception:
            shutil.copy2(image_path, out_path)
        rows.append({"image": str(out_path.relative_to(PROJECT_ROOT)), "text": build_caption(record, image_path.stem)})

    with captions_target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sample_path = PROJECT_ROOT / "data" / "samples" / "captions_sample.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in rows[:10]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
