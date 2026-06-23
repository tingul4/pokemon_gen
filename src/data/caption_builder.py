from __future__ import annotations

import unicodedata
from typing import Any

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
MAX_APPEARANCE_WORDS = 12
STYLE_TOKEN = "sks style"
ANCHOR_PHRASES = (
    STYLE_TOKEN,
    "single image",
    "single creature",
    "full body",
    "blank background",
    "clean composition",
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
    return (
        f"stats hp{stats['hp']} attack{stats['attack']} defense{stats['defense']} "
        f"special_attack{stats['special_attack']} special_defense{stats['special_defense']} speed{stats['speed']}"
    )


def _creature_text(text: str | None) -> str:
    return clean_text(text or "").replace("Pokemon", "creature")


def _type_phrase(types: list[str]) -> str:
    if not types:
        return "types elemental"
    return "types " + " ".join(clean_text(type_name).lower() for type_name in types if type_name)


def _limit_words(text: str, max_words: int) -> str:
    words = clean_text(text).split()
    return " ".join(words[:max_words])


def build_appearance_description(metadata: dict[str, Any] | None, fallback_name: str = "creature") -> str:
    appearance_bits = []
    classification = (metadata or {}).get("classification")
    description = (metadata or {}).get("description")
    if classification:
        appearance_bits.append(_creature_text(classification))
    if description:
        description_text = _creature_text(description)
        source_name = clean_text((metadata or {}).get("source_name") or (metadata or {}).get("name") or "")
        if source_name:
            description_text = description_text.replace(source_name, "this creature")
        appearance_bits.append(description_text)
    if appearance_bits:
        return " ".join(appearance_bits)
    return clean_text(fallback_name)


def build_label_text(
    *,
    types: list[str],
    stats: dict[str, int],
    appearance_description: str,
    max_appearance_words: int = MAX_APPEARANCE_WORDS,
) -> str:
    appearance = _limit_words(appearance_description, max_appearance_words)
    anchors = ", ".join(ANCHOR_PHRASES)
    return f"{anchors}, {_type_phrase(types)}, {_stat_phrase(stats)}, appearance {appearance}"


def build_caption(
    metadata: dict[str, Any] | None,
    fallback_name: str = "creature",
    appearance_description: str | None = None,
) -> str:
    types = (metadata or {}).get("types") or []
    if not types:
        types = ["elemental"]
    stats = (metadata or {}).get("stats") or {}
    return build_label_text(
        types=types,
        stats=stats,
        appearance_description=appearance_description or build_appearance_description(metadata, fallback_name),
    )
