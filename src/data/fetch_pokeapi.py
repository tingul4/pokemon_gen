from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.utils.config import ensure_dir


POKEAPI_BASE = "https://pokeapi.co/api/v2"
PREFERRED_FLAVOR_VERSIONS = [
    "violet",
    "scarlet",
    "legends-arceus",
    "shield",
    "sword",
    "ultra-moon",
    "ultra-sun",
    "moon",
    "sun",
    "alpha-sapphire",
    "omega-ruby",
    "y",
    "x",
    "white-2",
    "black-2",
    "white",
    "black",
    "soulsilver",
    "heartgold",
    "platinum",
    "pearl",
    "diamond",
    "leafgreen",
    "firered",
    "emerald",
    "sapphire",
    "ruby",
    "crystal",
    "silver",
    "gold",
    "yellow",
    "blue",
    "red",
]


def _get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace("\n", " ").replace("\f", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _english_value(items: list[dict[str, Any]], key: str) -> str | None:
    for item in items:
        if item.get("language", {}).get("name") == "en":
            return _clean_text(item.get(key))
    return None


def _species_profile(raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    entries = [
        item
        for item in raw.get("flavor_text_entries", [])
        if item.get("language", {}).get("name") == "en" and item.get("flavor_text")
    ]
    selected = None
    for version in PREFERRED_FLAVOR_VERSIONS:
        selected = next((item for item in entries if item.get("version", {}).get("name") == version), None)
        if selected:
            break
    if selected is None and entries:
        selected = entries[-1]

    return {
        "genus": _english_value(raw.get("genera", []), "genus"),
        "official_flavor_text": _clean_text(selected.get("flavor_text")) if selected else None,
        "flavor_version": selected.get("version", {}).get("name") if selected else None,
        "color": raw.get("color", {}).get("name") if raw.get("color") else None,
        "shape": raw.get("shape", {}).get("name") if raw.get("shape") else None,
        "habitat": raw.get("habitat", {}).get("name") if raw.get("habitat") else None,
        "is_baby": raw.get("is_baby"),
        "is_legendary": raw.get("is_legendary"),
        "is_mythical": raw.get("is_mythical"),
        "egg_groups": [item["name"] for item in raw.get("egg_groups", [])],
        "growth_rate": raw.get("growth_rate", {}).get("name") if raw.get("growth_rate") else None,
    }


def normalize_pokemon(raw: dict[str, Any], species_raw: dict[str, Any] | None = None) -> dict[str, Any]:
    stats = {item["stat"]["name"].replace("-", "_"): item["base_stat"] for item in raw["stats"]}
    normalized_stats = {
        "hp": stats["hp"],
        "attack": stats["attack"],
        "defense": stats["defense"],
        "special_attack": stats["special_attack"],
        "special_defense": stats["special_defense"],
        "speed": stats["speed"],
    }
    return {
        "id": raw["id"],
        "name": raw["name"],
        "types": [slot["type"]["name"] for slot in sorted(raw["types"], key=lambda item: item["slot"])],
        "stats": normalized_stats,
        "base_stat_total": sum(normalized_stats.values()),
        "height": raw.get("height"),
        "weight": raw.get("weight"),
        "abilities": [item["ability"]["name"] for item in raw.get("abilities", [])],
        "species_profile": _species_profile(species_raw),
    }


def fetch_pokemon_metadata(limit: int = 50, raw_dir: str | Path = "data/raw/pokeapi") -> list[dict[str, Any]]:
    raw_path = ensure_dir(raw_dir)
    index = _get_json(f"{POKEAPI_BASE}/pokemon?limit={limit}")
    records: list[dict[str, Any]] = []
    for item in tqdm(index["results"], desc="Fetching PokeAPI"):
        raw = _get_json(item["url"])
        species_raw = _get_json(raw["species"]["url"]) if raw.get("species", {}).get("url") else None
        with (raw_path / f"{raw['id']:04d}_{raw['name']}.json").open("w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2)
        if species_raw:
            species_dir = ensure_dir(raw_path / "species")
            with (species_dir / f"{raw['id']:04d}_{raw['name']}_species.json").open("w", encoding="utf-8") as handle:
                json.dump(species_raw, handle, indent=2)
        records.append(normalize_pokemon(raw, species_raw))
    return records


def save_metadata(records: list[dict[str, Any]], output_path: str | Path = "data/processed/metadata.json") -> Path:
    target = Path(output_path)
    if not target.is_absolute():
        target = Path.cwd() / target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    sample = target.parents[1] / "samples" / "pokeapi_sample.json"
    sample.parent.mkdir(parents=True, exist_ok=True)
    with sample.open("w", encoding="utf-8") as handle:
        json.dump(records[:5], handle, indent=2)
    return target
