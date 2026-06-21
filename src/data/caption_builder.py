from __future__ import annotations

from typing import Any

from src.generation.prompt_builder import type_hints


STAT_TRAITS = {
    "hp": "sturdy body",
    "attack": "strong claws or horns",
    "defense": "protective armor plates",
    "special_attack": "glowing elemental focus",
    "special_defense": "mystic defensive markings",
    "speed": "agile dynamic pose",
}


def _stat_phrase(stats: dict[str, int] | None) -> str:
    if not stats:
        return "balanced creature proportions"
    top_stats = sorted(stats.items(), key=lambda item: item[1], reverse=True)[:2]
    traits = [STAT_TRAITS.get(name, name.replace("_", " ")) for name, _ in top_stats]
    numeric = ", ".join(f"{name.replace('_', ' ')} {value}" for name, value in stats.items())
    return f"{' and '.join(traits)}, stat profile: {numeric}"


def _species_profile(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return (metadata or {}).get("species_profile") or {}


def _creature_genus(genus: str | None) -> str | None:
    if not genus:
        return None
    return genus.replace("Pokémon", "creature").replace("Pokemon", "creature")


def build_appearance_description(metadata: dict[str, Any] | None, fallback_name: str = "creature") -> str:
    types = (metadata or {}).get("types") or []
    if not types:
        types = ["elemental"]
    stats = (metadata or {}).get("stats") or {}
    height = (metadata or {}).get("height")
    weight = (metadata or {}).get("weight")
    abilities = (metadata or {}).get("abilities") or []
    species = _species_profile(metadata)
    colors, motifs = type_hints(types)
    type_text = f"{types[0]}-type" if len(types) == 1 else f"{types[0]} and {types[1]}-type"
    motif_text = ", ".join(motifs[:3]) if motifs else "expressive monster silhouette"
    color_text = ", ".join(colors[:3]) if colors else "bright balanced color palette"
    ability_text = ", ".join(str(item).replace("-", " ") for item in abilities[:2]) or "distinctive creature ability"
    size_bits = []
    if height is not None:
        size_bits.append(f"height index {height}")
    if weight is not None:
        size_bits.append(f"weight index {weight}")
    size_text = ", ".join(size_bits) if size_bits else "compact game-creature scale"

    official_bits = []
    genus = species.get("genus")
    if genus:
        official_bits.append(f"official genus: {genus}")
    if species.get("official_flavor_text"):
        version = species.get("flavor_version")
        version_text = f" ({version})" if version else ""
        official_bits.append(f"official Pokedex note{version_text}: {species['official_flavor_text']}")
    visual_metadata = []
    for key in ("color", "shape", "habitat"):
        if species.get(key):
            visual_metadata.append(f"{key}: {species[key]}")
    if visual_metadata:
        official_bits.append("official visual metadata: " + ", ".join(visual_metadata))
    official_text = "; ".join(official_bits)
    if official_text:
        official_text += "; "

    return (
        f"{fallback_name} PokeAPI profile: {official_text}"
        f"training visual summary: {type_text} creature with {motif_text}, {color_text}, "
        f"{_stat_phrase(stats)}, {ability_text} ability cues, {size_text}"
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
    motif_text = ", ".join(motifs[:2]) if motifs else "expressive monster silhouette"
    profile_bits = []
    genus = _creature_genus(species.get("genus"))
    if genus:
        profile_bits.append(genus.lower())
    if species.get("color"):
        profile_bits.append(str(species["color"]))
    if species.get("shape"):
        profile_bits.append(str(species["shape"]))
    profile_text = ", ".join(profile_bits[:3])
    color_text = profile_text or (", ".join(colors[:2]) if colors else "bright color palette")
    top_stats = sorted(stats.items(), key=lambda item: item[1], reverse=True)[:2]
    trait_text = ", ".join(STAT_TRAITS.get(name, name.replace("_", " ")) for name, _ in top_stats)
    stat_text = ""
    if stats:
        stat_text = (
            f"stats hp{stats['hp']} atk{stats['attack']} def{stats['defense']} "
            f"spa{stats['special_attack']} spd{stats['special_defense']} spe{stats['speed']}"
        )
    return (
        f"pokecreature_style, original {type_text}-type creature, {motif_text}, {color_text}, "
        f"{trait_text}, {stat_text}, "
        "clean game creature art"
    )
