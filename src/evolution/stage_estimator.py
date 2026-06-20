from __future__ import annotations

from typing import Mapping, TypedDict


REQUIRED_STATS = ("hp", "attack", "defense", "special_attack", "special_defense", "speed")


class StageEstimate(TypedDict):
    base_stat_total: int
    stage: str
    reason: str


def estimate_stage(stats: Mapping[str, int]) -> StageEstimate:
    missing = [name for name in REQUIRED_STATS if name not in stats]
    if missing:
        raise ValueError(f"Missing stats: {', '.join(missing)}")

    normalized: dict[str, int] = {}
    for name in REQUIRED_STATS:
        value = int(stats[name])
        if value < 0:
            raise ValueError(f"Stat {name} must be non-negative.")
        normalized[name] = value

    bst = sum(normalized.values())
    if bst <= 340:
        return {"base_stat_total": bst, "stage": "basic", "reason": "BST is 340 or lower."}
    if bst <= 480:
        return {"base_stat_total": bst, "stage": "stage_1", "reason": "BST is between 341 and 480."}
    return {"base_stat_total": bst, "stage": "stage_2", "reason": "BST is greater than 480."}

