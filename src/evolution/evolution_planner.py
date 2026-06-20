from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.evolution.stage_estimator import REQUIRED_STATS


STAGE_UP = {"basic": "stage_1", "stage_1": "stage_2"}
STAGE_DOWN = {"stage_2": "stage_1", "stage_1": "basic"}


def _scale_stats(stats: dict[str, int], factor: float) -> dict[str, int]:
    return {name: max(1, min(255, int(round(stats[name] * factor)))) for name in REQUIRED_STATS}


def plan_evolution(creature_metadata: dict[str, Any]) -> dict[str, Any]:
    stage = creature_metadata.get("stage")
    if stage not in STAGE_UP:
        raise ValueError("Only basic and stage_1 creatures can evolve.")
    planned = deepcopy(creature_metadata)
    planned["parent_id"] = creature_metadata.get("creature_id")
    planned["stage"] = STAGE_UP[stage]
    planned["stats"] = _scale_stats(creature_metadata["stats"], 1.20 if stage == "basic" else 1.15)
    motifs = ", ".join(creature_metadata.get("core_motifs", []))
    planned["appearance_description"] = (
        f"Evolved form preserving lineage motifs ({motifs}); larger silhouette, more confident pose, "
        "stronger elemental ornaments, and increased visual complexity."
    )
    return planned


def plan_devolution(creature_metadata: dict[str, Any]) -> dict[str, Any]:
    stage = creature_metadata.get("stage")
    if stage not in STAGE_DOWN:
        raise ValueError("Only stage_1 and stage_2 creatures can devolve.")
    planned = deepcopy(creature_metadata)
    planned["parent_id"] = creature_metadata.get("creature_id")
    planned["stage"] = STAGE_DOWN[stage]
    planned["stats"] = _scale_stats(creature_metadata["stats"], 0.80 if stage == "stage_1" else 0.85)
    motifs = ", ".join(creature_metadata.get("core_motifs", []))
    planned["appearance_description"] = (
        f"Previous form preserving lineage motifs ({motifs}); smaller body, softer silhouette, "
        "reduced ornamentation, and younger creature proportions."
    )
    return planned

