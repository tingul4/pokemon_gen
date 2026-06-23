from __future__ import annotations

from src.ui.state import cache_creature_stage, get_cached_stage, reset_lineage_state, stage_label


def test_stage_label_uses_traditional_chinese() -> None:
    assert stage_label("basic") == "基礎型"
    assert stage_label("stage_1") == "一階"
    assert stage_label("stage_2") == "二階"


def test_cache_creature_stage_reuses_existing_generation() -> None:
    session_state = {"stage_cache": {}, "current_creature": None, "lineage_id": None}
    creature = {"stage": "stage_1", "creature_id": "cr_stage_1", "image_path": "outputs/generations/stage1.png"}

    cache_creature_stage(session_state, creature)

    assert get_cached_stage(session_state, "stage_1") == creature


def test_reset_lineage_state_clears_stage_cache() -> None:
    session_state = {
        "stage_cache": {"stage_1": {"stage": "stage_1"}},
        "current_creature": {"stage": "stage_1"},
        "lineage_id": "ln_test",
    }

    reset_lineage_state(session_state)

    assert session_state["stage_cache"] == {}
    assert session_state["current_creature"] is None
    assert session_state["lineage_id"] is None
