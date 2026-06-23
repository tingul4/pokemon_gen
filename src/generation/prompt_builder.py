from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.data.caption_builder import build_label_text
from src.evolution.stage_estimator import REQUIRED_STATS
from src.utils.config import PROJECT_ROOT, load_yaml_config


STAT_TRAITS = {
    "hp": "bulky resilient body",
    "attack": "claws horns and strong limbs",
    "defense": "armor plates shield-like body parts",
    "special_attack": "glowing elemental aura and projectile motifs",
    "special_defense": "protective aura mystical markings crystal ornaments",
    "speed": "lean aerodynamic silhouette dynamic pose",
}
LORA_INFERENCE_APPEARANCE_WORDS = 16


def _compact(text: str, max_words: int = 16) -> str:
    words = text.replace("\n", " ").split()
    return " ".join(words[:max_words])


def _limit_words(parts: list[str], max_words: int = 41) -> str:
    selected: list[str] = []
    count = 0
    for part in parts:
        words = part.split()
        if not words:
            continue
        if count + len(words) > max_words:
            remaining = max_words - count
            if remaining > 0:
                selected.append(" ".join(words[:remaining]))
            break
        selected.append(part)
        count += len(words)
    return ", ".join(selected)


def _load_templates() -> dict[str, Any]:
    return load_yaml_config(PROJECT_ROOT / "configs" / "prompt_templates.yaml")


def type_hints(types: list[str]) -> tuple[list[str], list[str]]:
    templates = _load_templates()
    hint_map = templates.get("type_visual_hints", {})
    colors: list[str] = []
    motifs: list[str] = []
    for type_name in types:
        hint = hint_map.get(type_name, {})
        colors.extend(hint.get("colors", []))
        motifs.extend(hint.get("motifs", []))
    return list(dict.fromkeys(colors)), list(dict.fromkeys(motifs))


def stat_visual_traits(stats: Mapping[str, int]) -> list[str]:
    values = {name: int(stats.get(name, 0)) for name in REQUIRED_STATS}
    if not values:
        return []
    threshold = max(70, int(sum(values.values()) / len(values)))
    return [STAT_TRAITS[name] for name, value in values.items() if value >= threshold]


def compact_stats_text(stats: Mapping[str, int]) -> str:
    values = {name: int(stats.get(name, 0)) for name in REQUIRED_STATS}
    if not values:
        return ""
    return (
        f"stats hp{values['hp']} attack{values['attack']} defense{values['defense']} "
        f"special_attack{values['special_attack']} special_defense{values['special_defense']} speed{values['speed']}"
    )


def build_negative_prompt(extra: str | None = None) -> str:
    templates = _load_templates()
    parts = list(templates.get("negative_prompt", []))
    deduped = list(dict.fromkeys(part.strip() for part in parts if part))
    return _limit_words(deduped, max_words=42)


def build_sdxl_prompt(
    *,
    types: list[str],
    stats: Mapping[str, int],
    appearance_description: str,
    llm_prompt: str | None = None,
    color_palette: list[str] | None = None,
    core_motifs: list[str] | None = None,
    use_lora: bool = False,
) -> str:
    colors, type_motifs = type_hints(types)
    if use_lora:
        return build_label_text(
            types=types,
            stats={name: int(stats.get(name, 0)) for name in REQUIRED_STATS},
            appearance_description=appearance_description or llm_prompt or "",
            max_appearance_words=LORA_INFERENCE_APPEARANCE_WORDS,
        )

    motifs = list(type_motifs)
    if color_palette:
        colors = list(dict.fromkeys(color_palette + colors))
    if core_motifs:
        motifs = list(dict.fromkeys(motifs + core_motifs))

    type_text = " and ".join(types)
    parts = [
        f"{type_text} type creature" if type_text else "fantasy elemental creature",
        "single front view",
        "full body",
        "creature concept art",
        "simple background",
    ]
    if type_motifs:
        parts.extend(type_motifs[:3])
    if appearance_description:
        parts.append(_compact(appearance_description, 13))
    remaining_motifs = [motif for motif in motifs if motif not in type_motifs]
    if remaining_motifs:
        parts.extend(remaining_motifs[:2])
    parts.extend(stat_visual_traits(stats)[:1])
    if colors:
        parts.append(f"{', '.join(colors[:3])} color palette")
    if llm_prompt:
        parts.append(_compact(llm_prompt, 8))
    deduped = list(dict.fromkeys(part.strip() for part in parts if part))
    return _limit_words(deduped)
