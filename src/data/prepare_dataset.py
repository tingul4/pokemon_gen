from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from src.data.caption_builder import build_appearance_description, build_caption, clean_json_value, clean_text
from src.utils.config import PROJECT_ROOT, ensure_dir


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DATASET_NAME_ALIASES = {
    "farfetchd": "farfetchd",
    "sirfetchd": "sirfetchd",
    "mrmime": "mr-mime",
    "mrrime": "mr-rime",
    "mimejr": "mime-jr",
    "porygonz": "porygon-z",
    "hooh": "ho-oh",
    "hakamoo": "hakamo-o",
    "jangmoo": "jangmo-o",
    "kommoo": "kommo-o",
    "tapubulu": "tapu-bulu",
    "tapufini": "tapu-fini",
    "tapukoko": "tapu-koko",
    "tapulele": "tapu-lele",
    "typenull": "type-null",
    "flabb": "flabebe",
    "aegislash": "aegislash-shield",
    "basculin": "basculin-red-striped",
    "darmanitan": "darmanitan-standard",
    "deoxys": "deoxys-normal",
    "eiscue": "eiscue-ice",
    "frillish": "frillish-male",
    "giratina": "giratina-altered",
    "gourgeist": "gourgeist-average",
    "indeedee": "indeedee-male",
    "jellicent": "jellicent-male",
    "keldeo": "keldeo-ordinary",
    "landorus": "landorus-incarnate",
    "lycanroc": "lycanroc-midday",
    "meloetta": "meloetta-aria",
    "meowstic": "meowstic-male",
    "mimikyu": "mimikyu-disguised",
    "minior": "minior-red-meteor",
    "morpeko": "morpeko-full-belly",
    "oricorio": "oricorio-baile",
    "pumpkaboo": "pumpkaboo-average",
    "pyroar": "pyroar-male",
    "shaymin": "shaymin-land",
    "thundurus": "thundurus-incarnate",
    "tornadus": "tornadus-incarnate",
    "toxtricity": "toxtricity-amped",
    "urshifu": "urshifu-single-strike",
    "wishiwashi": "wishiwashi-solo",
    "wormadam": "wormadam-plant",
    "zygarde": "zygarde-50",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def dataset_name_to_pokeapi_name(name: str) -> str:
    if "♀" in name or "ΓÖÇ" in name:
        return "nidoran-f"
    if "♂" in name or "ΓÖé" in name:
        return "nidoran-m"
    slug = _slug(name)
    return DATASET_NAME_ALIASES.get(slug, re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"))


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
        metadata[record["name"]] = record
        metadata[_slug(record["name"])] = record
        metadata[str(record["id"])] = record
    return metadata


def _match_metadata(image_path: Path, metadata: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    dataset_name = image_path.parent.name
    canonical = dataset_name_to_pokeapi_name(dataset_name)
    if canonical in metadata:
        return metadata[canonical]
    canonical_slug = _slug(canonical)
    if canonical_slug in metadata:
        return metadata[canonical_slug]

    stem = _slug(image_path.stem)
    numeric = re.match(r"^(\d+)", stem)
    if numeric and numeric.group(1) in metadata:
        return metadata[numeric.group(1)]
    for key, record in metadata.items():
        if not key.isdigit() and key and (key == stem or key in stem):
            return record
    return None


def _annotation_row(
    image_path: Path,
    processed_image_path: Path,
    metadata: dict[str, Any] | None,
    caption: str,
    appearance_description: str,
) -> dict[str, Any]:
    source_name = image_path.parent.name
    clean_species_profile = clean_json_value((metadata or {}).get("species_profile") or {})
    return {
        "source_image": str(image_path.relative_to(PROJECT_ROOT) if image_path.is_relative_to(PROJECT_ROOT) else image_path),
        "image": str(processed_image_path.relative_to(PROJECT_ROOT)),
        "source_name": clean_text(source_name),
        "pokeapi_name": (metadata or {}).get("name") or dataset_name_to_pokeapi_name(source_name),
        "pokemon_id": (metadata or {}).get("id"),
        "label": {
            "types": (metadata or {}).get("types") or [],
            "stats": (metadata or {}).get("stats") or {},
            "base_stat_total": (metadata or {}).get("base_stat_total"),
            "height": (metadata or {}).get("height"),
            "weight": (metadata or {}).get("weight"),
            "abilities": (metadata or {}).get("abilities") or [],
            "species_profile": clean_species_profile,
            "appearance_description": appearance_description,
        },
        "caption": caption,
    }


def prepare_lora_dataset(
    raw_image_dir: str | Path = "data/raw/kaggle_pokemon_image_dataset/images",
    output_image_dir: str | Path = "data/processed/lora_images",
    captions_path: str | Path = "data/processed/captions.jsonl",
    annotations_path: str | Path = "data/processed/annotations.jsonl",
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
    annotations_target = Path(annotations_path)
    if not annotations_target.is_absolute():
        annotations_target = PROJECT_ROOT / annotations_target
    annotations_target.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    annotations: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for idx, image_path in enumerate(tqdm(images, desc="Preparing LoRA images")):
        record = _match_metadata(image_path, metadata)
        if record is None:
            unmatched.append(str(image_path.parent.name))
        appearance = build_appearance_description(record, image_path.parent.name)
        caption = build_caption(record, image_path.parent.name, appearance)
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
        rows.append({"image": str(out_path.relative_to(PROJECT_ROOT)), "text": caption})
        annotations.append(_annotation_row(image_path, out_path, record, caption, appearance))

    with captions_target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with annotations_target.open("w", encoding="utf-8") as handle:
        for row in annotations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sample_path = PROJECT_ROOT / "data" / "samples" / "captions_sample.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in rows[:10]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    annotation_sample_path = PROJECT_ROOT / "data" / "samples" / "annotations_sample.jsonl"
    with annotation_sample_path.open("w", encoding="utf-8") as handle:
        for row in annotations[:10]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if unmatched:
        unmatched_path = PROJECT_ROOT / "data" / "processed" / "unmatched_names.json"
        unmatched_path.write_text(json.dumps(sorted(set(unmatched)), indent=2), encoding="utf-8")
    return rows
