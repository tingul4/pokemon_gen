from __future__ import annotations

from src.generation.prompt_builder import build_negative_prompt, build_sdxl_prompt, compact_stats_text, stat_visual_traits


def test_prompt_includes_types_and_lora_token() -> None:
    prompt = build_sdxl_prompt(
        types=["fire", "flying"],
        stats={"hp": 70, "attack": 95, "defense": 70, "special_attack": 110, "special_defense": 75, "speed": 120},
        appearance_description="feathered wings and a glowing flame crest",
        use_lora=True,
    )
    assert prompt.startswith("pokecreature_style, original fire-flying-type creature")
    assert "feathered wings" in prompt
    assert "single front view" in prompt
    assert "stats hp70 atk95 def70 spa110 spd75 spe120" in prompt
    assert prompt.endswith("clean game creature art")


def test_non_lora_prompt_includes_single_front_view() -> None:
    prompt = build_sdxl_prompt(
        types=["fire", "flying"],
        stats={"hp": 70, "attack": 95, "defense": 70, "special_attack": 110, "special_defense": 75, "speed": 120},
        appearance_description="feathered wings and a glowing flame crest",
        use_lora=False,
    )
    assert "fire and flying type creature" in prompt
    assert "single front view" in prompt


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
    assert "multiple views" in negative
    assert "multiple poses" in negative
    assert "reference sheet" in negative
    assert "turnaround" in negative


def test_negative_prompt_merges_llm_prompt_with_composition_guards() -> None:
    negative = build_negative_prompt("ugly, blurry")
    assert "ugly" in negative
    assert "character sheet" in negative
    assert "front side back views" in negative


def test_lora_prompt_matches_caption_jsonl_shape() -> None:
    prompt = build_sdxl_prompt(
        types=["electric"],
        stats={"hp": 70, "attack": 70, "defense": 102, "special_attack": 70, "special_defense": 141, "speed": 32},
        appearance_description=(
            "A squat, heavily armored amphibian, blending features of a turtle and a frog. "
            "Its back is covered by a thick, segmented, dark moss-green shell. "
            "Its skin is robust and leathery, adorned with glowing electric blue and vibrant yellow crystalline growths."
        ),
        llm_prompt=(
            "A sturdy, squat creature resembling a hybrid of a turtle and a frog with glowing electric blue "
            "and vibrant yellow crystalline growths and static energy."
        ),
        color_palette=["Deep moss green and earthy browns", "Electric blues, vibrant yellows"],
        core_motifs=["Armored amphibian", "Electrical insulation", "Energy absorption", "Slow resilience", "Living battery"],
        use_lora=True,
    )
    assert prompt.startswith("pokecreature_style, original electric-type creature")
    assert "turtle and a frog" in prompt
    assert "dark moss-green shell" in prompt
    assert "single front view" in prompt
    assert "stats hp70 atk70 def102 spa70 spd141 spe32" in prompt
    assert prompt.endswith("clean game creature art")


def test_high_speed_trait() -> None:
    traits = stat_visual_traits(
        {"hp": 40, "attack": 40, "defense": 40, "special_attack": 40, "special_defense": 40, "speed": 120}
    )
    assert any("aerodynamic" in trait for trait in traits)
