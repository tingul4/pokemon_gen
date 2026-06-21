from __future__ import annotations

import unicodedata
from typing import Any

from src.generation.prompt_builder import type_hints


TEXT_REPLACEMENTS = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "—": " - ",
        "–": "-",
        "…": "...",
    }
)


def clean_text(text: Any) -> str:
    value = str(text).translate(TEXT_REPLACEMENTS)
    value = value.replace("POKéMON", "Pokemon").replace("Pokémon", "Pokemon")
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(normalized.split())


def clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    return value


def _stat_phrase(stats: dict[str, int] | None) -> str:
    if not stats:
        return "stats unavailable"
    numeric = ", ".join(f"{name.replace('_', ' ')} {value}" for name, value in stats.items())
    return f"stat profile: {numeric}"


def _species_profile(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return (metadata or {}).get("species_profile") or {}


def _creature_genus(genus: str | None) -> str | None:
    if not genus:
        return None
    return clean_text(genus).replace("Pokemon", "creature")


def build_appearance_description(metadata: dict[str, Any] | None, fallback_name: str = "creature") -> str:
    types = (metadata or {}).get("types") or []
    if not types:
        types = ["elemental"]
    stats = (metadata or {}).get("stats") or {}
    height = (metadata or {}).get("height")
    weight = (metadata or {}).get("weight")
    abilities = (metadata or {}).get("abilities") or []
    species = _species_profile(metadata)
    type_text = f"{types[0]}-type" if len(types) == 1 else f"{types[0]} and {types[1]}-type"
    ability_text = ", ".join(clean_text(str(item).replace("-", " ")) for item in abilities[:2]) or "distinctive creature ability"
    size_bits = []
    if height is not None:
        size_bits.append(f"height index {height}")
    if weight is not None:
        size_bits.append(f"weight index {weight}")
    size_text = ", ".join(size_bits) if size_bits else "compact game-creature scale"

    official_bits = []
    genus = species.get("genus")
    if genus:
        official_bits.append(f"official genus: {clean_text(genus)}")
    if species.get("official_flavor_text"):
        version = species.get("flavor_version")
        version_text = f" ({version})" if version else ""
        official_bits.append(f"official Pokedex note{version_text}: {clean_text(species['official_flavor_text'])}")
    visual_metadata = []
    for key in ("color", "shape", "habitat"):
        if species.get(key):
            visual_metadata.append(f"{key}: {clean_text(species[key])}")
    if visual_metadata:
        official_bits.append("official visual metadata: " + ", ".join(visual_metadata))
    official_text = "; ".join(official_bits)
    if official_text:
        official_text += "; "

    return (
        f"{clean_text(fallback_name)} PokeAPI profile: {official_text}"
        f"conditioning summary: {type_text} creature, {_stat_phrase(stats)}, "
        f"{ability_text} ability cues, {size_text}"
    )


def build_caption(
    metadata: dict[str, Any] | None,
    fallback_name: str = "creature",
    appearance_description: str | None = None,
) -> str:
    types = (metadata or {}).get("types") or []
    if not types:
        types = ["elemental"]
    stats = (metadata or {}).get("stats") or {}
    colors, motifs = type_hints(types)
    species = _species_profile(metadata)
    type_text = "-".join(types[:2])
    profile_bits = []
    genus = _creature_genus(species.get("genus"))
    if genus:
        profile_bits.append(genus.lower())
    if species.get("color"):
        profile_bits.append(clean_text(species["color"]))
    if species.get("shape"):
        profile_bits.append(clean_text(species["shape"]))
    profile_text = ", ".join(profile_bits[:3])
    if profile_text:
        descriptor_text = profile_text
    else:
        fallback_bits = motifs[:1] + colors[:1]
        descriptor_text = ", ".join(fallback_bits) if fallback_bits else "creature profile"
    stat_text = ""
    if stats:
        stat_text = (
            f"stats hp{stats['hp']} atk{stats['attack']} def{stats['defense']} "
            f"spa{stats['special_attack']} spd{stats['special_defense']} spe{stats['speed']}"
        )
    return (
        f"pokecreature_style, original {type_text}-type creature, {descriptor_text}, "
        f"{stat_text}, "
        "clean game creature art"
    )
