from __future__ import annotations

from src.generation.prompt_builder import build_negative_prompt, build_sdxl_prompt, compact_stats_text, stat_visual_traits


def test_lora_prompt_includes_types_stats_and_appearance() -> None:
    prompt = build_sdxl_prompt(
        types=["fire", "flying"],
        stats={"hp": 70, "attack": 95, "defense": 70, "special_attack": 110, "special_defense": 75, "speed": 120},
        appearance_description="feathered wings and a glowing flame crest",
        use_lora=True,
    )
    assert prompt.startswith("sks style, single image, single creature, full body, blank background, clean composition")
    assert "types fire flying, stats hp70 attack95 defense70" in prompt
    assert "feathered wings" in prompt
    assert "special_attack110 special_defense75 speed120" in prompt
    assert "appearance feathered wings and a glowing flame crest" in prompt


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
        == "stats hp39 attack52 defense43 special_attack60 special_defense50 speed65"
    )


def test_negative_prompt_keeps_only_minimal_layout_controls() -> None:
    negative = build_negative_prompt()
    assert negative == "multiple views, multi panel, grid layout, reference sheet"
    assert "official pokemon" not in negative
    assert "pikachu" not in negative
    assert "copyrighted character" not in negative
    assert "low quality" not in negative


def test_negative_prompt_ignores_llm_extra_terms() -> None:
    negative = build_negative_prompt("ugly, blurry")
    assert "multiple views" in negative
    assert "ugly" not in negative
    assert "blurry" not in negative


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
    assert prompt.startswith("sks style, single image, single creature, full body, blank background, clean composition")
    assert "types electric, stats hp70 attack70 defense102" in prompt
    assert "features of a turtle and a" in prompt
    assert "frog" not in prompt
    assert "Its back is covered by" not in prompt
    assert "dark moss-green shell" not in prompt
    assert "special_attack70 special_defense141 speed32" in prompt
    assert "appearance A squat" in prompt


def test_lora_prompt_limits_long_appearance_description() -> None:
    prompt = build_sdxl_prompt(
        types=["water"],
        stats={"hp": 70, "attack": 70, "defense": 102, "special_attack": 70, "special_defense": 141, "speed": 32},
        appearance_description=" ".join(f"detail{i}" for i in range(60)),
        use_lora=True,
    )
    assert "detail11" in prompt
    assert "detail12" not in prompt


def test_high_speed_trait() -> None:
    traits = stat_visual_traits(
        {"hp": 40, "attack": 40, "defense": 40, "special_attack": 40, "special_defense": 40, "speed": 120}
    )
    assert any("aerodynamic" in trait for trait in traits)
