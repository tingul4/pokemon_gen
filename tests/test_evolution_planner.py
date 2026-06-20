from __future__ import annotations

import pytest

from src.evolution.evolution_planner import plan_devolution, plan_evolution


CREATURE = {
    "creature_id": "cr_test",
    "stage": "basic",
    "types": ["grass"],
    "stats": {"hp": 50, "attack": 50, "defense": 50, "special_attack": 50, "special_defense": 50, "speed": 50},
    "core_motifs": ["leaf shell"],
    "color_palette": ["green"],
}


def test_evolution_advances_stage_and_stats() -> None:
    evolved = plan_evolution(CREATURE)
    assert evolved["stage"] == "stage_1"
    assert evolved["stats"]["hp"] > CREATURE["stats"]["hp"]


def test_devolution_reduces_stage_and_stats() -> None:
    stage_two = dict(CREATURE, stage="stage_2")
    devolved = plan_devolution(stage_two)
    assert devolved["stage"] == "stage_1"
    assert devolved["stats"]["hp"] < stage_two["stats"]["hp"]


def test_stage_two_cannot_evolve() -> None:
    with pytest.raises(ValueError):
        plan_evolution(dict(CREATURE, stage="stage_2"))

