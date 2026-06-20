from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.utils.config import ensure_dir


POKEAPI_BASE = "https://pokeapi.co/api/v2"


def _get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_pokemon(raw: dict[str, Any]) -> dict[str, Any]:
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
    }


def fetch_pokemon_metadata(limit: int = 50, raw_dir: str | Path = "data/raw/pokeapi") -> list[dict[str, Any]]:
    raw_path = ensure_dir(raw_dir)
    index = _get_json(f"{POKEAPI_BASE}/pokemon?limit={limit}")
    records: list[dict[str, Any]] = []
    for item in tqdm(index["results"], desc="Fetching PokeAPI"):
        raw = _get_json(item["url"])
        with (raw_path / f"{raw['id']:04d}_{raw['name']}.json").open("w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2)
        records.append(normalize_pokemon(raw))
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

