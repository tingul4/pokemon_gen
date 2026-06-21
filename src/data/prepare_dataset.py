from __future__ import annotations

import csv
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
STAT_COLUMNS = {
    "hp": "hp",
    "attack": "attack",
    "defense": "defense",
    "sp_attack": "special_attack",
    "sp_defense": "special_defense",
    "speed": "speed",
}


def _project_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else PROJECT_ROOT / target


def _to_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(str(value).strip()))


def _to_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(str(value).strip())


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _clean_name(value: Any) -> str:
    text = str(value or "")
    text = text.replace("♀", " Female").replace("♂", " Male")
    return clean_text(text)


def _image_number(path: Path) -> int | None:
    match = re.match(r"^(\d{3})", path.stem)
    return int(match.group(1)) if match else None


def _form_name(path: Path, fallback: str) -> str:
    suffix = re.sub(r"^\d{3}", "", path.stem).strip("-_ ")
    if not suffix:
        return fallback
    return clean_text(suffix.replace("_", " "))


def _read_pokedex_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    target = _project_path(csv_path)
    if not target.exists():
        raise FileNotFoundError(f"Missing pokedex CSV: {target}")
    with target.open("r", encoding="utf-16", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [_normalise_csv_row(row) for row in reader]


def _normalise_csv_row(row: dict[str, str]) -> dict[str, Any]:
    national_number = _to_int(row.get("national_number"))
    name = _clean_name(row.get("english_name") or f"pokemon-{national_number}")
    types = [clean_text(row.get("primary_type", "")).lower()]
    secondary = clean_text(row.get("secondary_type", "")).lower()
    if secondary:
        types.append(secondary)
    stats = {
        output_name: _to_int(row.get(input_name)) or 0
        for input_name, output_name in STAT_COLUMNS.items()
    }
    abilities = [
        clean_text(row.get(key, ""))
        for key in ("abilities_0", "abilities_1", "abilities_2", "abilities_hidden")
        if clean_text(row.get(key, ""))
    ]
    evolution_chain = [
        clean_text(row.get(f"evochain_{idx}", ""))
        for idx in range(7)
        if clean_text(row.get(f"evochain_{idx}", ""))
    ]
    return {
        "id": national_number,
        "national_number": national_number,
        "name": name,
        "source_name": name,
        "generation": clean_text(row.get("gen", "")),
        "types": types,
        "classification": clean_text(row.get("classification", "")).replace("Pokemon", "creature"),
        "description": clean_text(row.get("description", "")).replace("Pokemon", "creature"),
        "stats": stats,
        "base_stat_total": sum(stats.values()),
        "height_m": _to_float(row.get("height_m")),
        "weight_kg": _to_float(row.get("weight_kg")),
        "abilities": abilities,
        "is_sublegendary": _flag(row.get("is_sublegendary")),
        "is_legendary": _flag(row.get("is_legendary")),
        "is_mythical": _flag(row.get("is_mythical")),
        "evolution_chain": evolution_chain,
        "forms": {
            "gigantamax": clean_text(row.get("gigantamax", "")),
            "mega_evolution": clean_text(row.get("mega_evolution", "")),
            "mega_evolution_alt": clean_text(row.get("mega_evolution_alt", "")),
        },
    }


def load_metadata(path: str | Path = "data/processed/metadata.json") -> dict[int, dict[str, Any]]:
    target = _project_path(path)
    if not target.exists():
        return {}
    records = json.loads(target.read_text(encoding="utf-8"))
    return {int(record["national_number"]): record for record in records if record.get("national_number") is not None}


def _write_metadata(records: list[dict[str, Any]], output_path: str | Path) -> None:
    target = _project_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    sample_path = PROJECT_ROOT / "data" / "samples" / "metadata_sample.json"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(records[:10], indent=2, ensure_ascii=False), encoding="utf-8")


def _dataset_root(raw_dir: str | Path) -> Path:
    root = _project_path(raw_dir)
    if (root / "data" / "pokemon.csv").exists() and (root / "images").exists():
        return root
    if root.name == "images" and (root.parent / "data" / "pokemon.csv").exists():
        return root.parent
    raise FileNotFoundError(
        "Expected cristobalmitchell/pokedex layout with data/pokemon.csv and images/ under "
        f"{root}"
    )


def find_images(raw_dir: str | Path) -> list[Path]:
    root = _dataset_root(raw_dir)
    image_root = root / "images"
    search_dirs = [image_root / "large_images", image_root / "alt_images"]
    images: list[Path] = []
    for search_dir in search_dirs:
        if search_dir.exists():
            images.extend(path for path in search_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    return sorted(images, key=lambda path: (path.parent.name, path.name))


def _annotation_row(
    image_path: Path,
    processed_image_path: Path,
    metadata: dict[str, Any],
    caption: str,
    appearance_description: str,
) -> dict[str, Any]:
    form_name = _form_name(image_path, metadata["name"])
    source_name = metadata["name"] if form_name == metadata["name"] else f"{metadata['name']} {form_name}"
    return {
        "source_image": str(image_path.relative_to(PROJECT_ROOT) if image_path.is_relative_to(PROJECT_ROOT) else image_path),
        "image": str(processed_image_path.relative_to(PROJECT_ROOT)),
        "source_name": source_name,
        "pokemon_name": metadata["name"],
        "national_number": metadata["national_number"],
        "form": form_name,
        "label": {
            "types": metadata["types"],
            "stats": metadata["stats"],
            "base_stat_total": metadata["base_stat_total"],
            "height_m": metadata["height_m"],
            "weight_kg": metadata["weight_kg"],
            "abilities": metadata["abilities"],
            "classification": metadata["classification"],
            "description": metadata["description"],
            "generation": metadata["generation"],
            "is_sublegendary": metadata["is_sublegendary"],
            "is_legendary": metadata["is_legendary"],
            "is_mythical": metadata["is_mythical"],
            "evolution_chain": metadata["evolution_chain"],
            "forms": metadata["forms"],
            "appearance_description": appearance_description,
        },
        "caption": caption,
    }


def prepare_lora_dataset(
    raw_image_dir: str | Path = "data/raw/cristobalmitchell_pokedex",
    output_image_dir: str | Path = "data/processed/lora_images",
    captions_path: str | Path = "data/processed/captions.jsonl",
    annotations_path: str | Path = "data/processed/annotations.jsonl",
    metadata_path: str | Path = "data/processed/metadata.json",
    max_images: int | None = None,
    resolution: int = 768,
) -> list[dict[str, str]]:
    root = _dataset_root(raw_image_dir)
    metadata_records = _read_pokedex_csv(root / "data" / "pokemon.csv")
    _write_metadata(metadata_records, metadata_path)
    metadata = {int(record["national_number"]): record for record in metadata_records}
    images = find_images(root)
    if max_images:
        images = images[:max_images]

    output_dir = ensure_dir(output_image_dir)
    captions_target = _project_path(captions_path)
    captions_target.parent.mkdir(parents=True, exist_ok=True)
    annotations_target = _project_path(annotations_path)
    annotations_target.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    annotations: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for idx, image_path in enumerate(tqdm(images, desc="Preparing LoRA images")):
        number = _image_number(image_path)
        record = metadata.get(number) if number is not None else None
        if record is None:
            unmatched.append(image_path.name)
            continue
        appearance = build_appearance_description(record, _form_name(image_path, record["name"]))
        caption = build_caption(record, _form_name(image_path, record["name"]), appearance)
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
            handle.write(json.dumps(clean_json_value(row), ensure_ascii=False) + "\n")

    sample_path = PROJECT_ROOT / "data" / "samples" / "captions_sample.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in rows[:10]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    annotation_sample_path = PROJECT_ROOT / "data" / "samples" / "annotations_sample.jsonl"
    with annotation_sample_path.open("w", encoding="utf-8") as handle:
        for row in annotations[:10]:
            handle.write(json.dumps(clean_json_value(row), ensure_ascii=False) + "\n")
    unmatched_path = PROJECT_ROOT / "data" / "processed" / "unmatched_names.json"
    if unmatched:
        unmatched_path.write_text(json.dumps(sorted(set(unmatched)), indent=2), encoding="utf-8")
    elif unmatched_path.exists():
        unmatched_path.unlink()
    return rows
