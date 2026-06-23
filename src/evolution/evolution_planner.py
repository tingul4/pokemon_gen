from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from src.evolution.stage_estimator import REQUIRED_STATS


STAGE_UP = {"basic": "stage_1", "stage_1": "stage_2"}
STAGE_DOWN = {"stage_2": "stage_1", "stage_1": "basic"}


def _scale_stats(stats: dict[str, int], factor: float) -> dict[str, int]:
    return {name: max(1, min(255, int(round(stats[name] * factor)))) for name in REQUIRED_STATS}


def _motif_text(creature_metadata: dict[str, Any], limit: int = 2) -> str:
    motifs: list[str] = []
    for motif in creature_metadata.get("core_motifs", []):
        motif_title = str(motif).split(":", 1)[0]
        words = re.sub(r"[^A-Za-z0-9 ]+", " ", motif_title).split()
        if words:
            motifs.append(" ".join(words[:3]))
    return ", ".join(motifs[:limit])


def plan_evolution(creature_metadata: dict[str, Any]) -> dict[str, Any]:
    stage = creature_metadata.get("stage")
    if stage not in STAGE_UP:
        raise ValueError("Only basic and stage_1 creatures can evolve.")
    planned = deepcopy(creature_metadata)
    planned["parent_id"] = creature_metadata.get("creature_id")
    planned["stage"] = STAGE_UP[stage]
    planned["stats"] = _scale_stats(creature_metadata["stats"], 1.20 if stage == "basic" else 1.15)
    motifs = _motif_text(creature_metadata)
    motif_phrase = f" with {motifs}" if motifs else ""
    planned["appearance_description"] = f"larger evolved creature{motif_phrase}, confident pose, stronger elemental ornaments"
    return planned


def plan_devolution(creature_metadata: dict[str, Any]) -> dict[str, Any]:
    stage = creature_metadata.get("stage")
    if stage not in STAGE_DOWN:
        raise ValueError("Only stage_1 and stage_2 creatures can devolve.")
    planned = deepcopy(creature_metadata)
    planned["parent_id"] = creature_metadata.get("creature_id")
    planned["stage"] = STAGE_DOWN[stage]
    planned["stats"] = _scale_stats(creature_metadata["stats"], 0.80 if stage == "stage_1" else 0.85)
    motifs = _motif_text(creature_metadata)
    motif_phrase = f" with {motifs}" if motifs else ""
    planned["appearance_description"] = f"younger previous creature{motif_phrase}, smaller body, softer silhouette, simple ornaments"
    return planned
