from __future__ import annotations

from typing import Any


STAGE_LABELS = {
    "basic": "基礎型",
    "stage_1": "一階",
    "stage_2": "二階",
}


def default_session_state() -> dict[str, Any]:
    return {
        "current_creature": None,
        "lineage_id": None,
        "last_result": None,
        "stage_cache": {},
    }


def stage_label(stage: str | None) -> str:
    if not stage:
        return "未生成"
    return STAGE_LABELS.get(stage, stage)


def reset_lineage_state(session_state: dict[str, Any]) -> None:
    session_state["current_creature"] = None
    session_state["lineage_id"] = None
    session_state["stage_cache"] = {}


def cache_creature_stage(session_state: dict[str, Any], creature: dict[str, Any]) -> None:
    stage = creature.get("stage")
    if stage not in STAGE_LABELS:
        return
    cache = dict(session_state.get("stage_cache") or {})
    cache[stage] = creature
    session_state["stage_cache"] = cache


def get_cached_stage(session_state: dict[str, Any], stage: str) -> dict[str, Any] | None:
    cached = session_state.get("stage_cache", {}).get(stage)
    return cached if isinstance(cached, dict) else None
