from __future__ import annotations

import pytest

from src.evolution.stage_estimator import estimate_stage


BASE = {"hp": 50, "attack": 50, "defense": 50, "special_attack": 50, "special_defense": 50, "speed": 50}


def test_basic_boundary() -> None:
    stats = dict(BASE, hp=90)
    assert sum(stats.values()) == 340
    assert estimate_stage(stats)["stage"] == "basic"


def test_stage_one_lower_boundary() -> None:
    stats = dict(BASE, hp=91)
    assert estimate_stage(stats)["stage"] == "stage_1"


def test_stage_one_upper_boundary() -> None:
    stats = {"hp": 80, "attack": 80, "defense": 80, "special_attack": 80, "special_defense": 80, "speed": 80}
    assert estimate_stage(stats)["stage"] == "stage_1"


def test_stage_two_boundary() -> None:
    stats = {"hp": 81, "attack": 80, "defense": 80, "special_attack": 80, "special_defense": 80, "speed": 80}
    assert estimate_stage(stats)["stage"] == "stage_2"


def test_negative_stat_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_stage(dict(BASE, hp=-1))


def test_missing_stat_rejected() -> None:
    incomplete = dict(BASE)
    incomplete.pop("speed")
    with pytest.raises(ValueError):
        estimate_stage(incomplete)

