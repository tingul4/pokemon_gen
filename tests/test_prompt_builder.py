from __future__ import annotations

from src.generation.prompt_builder import build_negative_prompt, build_sdxl_prompt, compact_stats_text, stat_visual_traits


def test_prompt_includes_types_and_lora_token() -> None:
    prompt = build_sdxl_prompt(
        types=["fire", "flying"],
        stats={"hp": 70, "attack": 95, "defense": 70, "special_attack": 110, "special_defense": 75, "speed": 120},
        appearance_description="feathered wings and a glowing flame crest",
        use_lora=True,
    )
    assert "pokecreature_style" in prompt
    assert "fire and flying type creature" in prompt
    assert "feathered wings" in prompt
    assert "stats hp70 atk95 def70 spa110 spd75 spe120" in prompt


def test_compact_stats_text_matches_lora_caption_format() -> None:
    assert (
        compact_stats_text(
            {"hp": 39, "attack": 52, "defense": 43, "special_attack": 60, "special_defense": 50, "speed": 65}
        )
        == "stats hp39 atk52 def43 spa60 spd50 spe65"
    )


def test_negative_prompt_blocks_official_names() -> None:
    negative = build_negative_prompt()
    assert "pikachu" in negative
    assert "copyrighted character" in negative


def test_high_speed_trait() -> None:
    traits = stat_visual_traits(
        {"hp": 40, "attack": 40, "defense": 40, "special_attack": 40, "special_defense": 40, "speed": 120}
    )
    assert any("aerodynamic" in trait for trait in traits)
