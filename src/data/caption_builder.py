from __future__ import annotations

from typing import Any

from src.generation.prompt_builder import type_hints


def build_caption(metadata: dict[str, Any] | None, fallback_name: str = "creature") -> str:
    types = (metadata or {}).get("types") or []
    if not types:
        types = ["elemental"]
    colors, motifs = type_hints(types)
    type_text = f"{types[0]} creature" if len(types) == 1 else f"{types[0]} and {types[1]} creature"
    motif_text = ", ".join(motifs[:2]) if motifs else "expressive monster silhouette"
    color_text = ", ".join(colors[:3]) if colors else "bright balanced color palette"
    return (
        f"pokecreature_style, an original {type_text}, {motif_text}, {color_text}, "
        "cute monster illustration, clean line art, game creature concept art"
    )

