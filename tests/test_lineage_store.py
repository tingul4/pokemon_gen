from __future__ import annotations

from src.evolution.lineage_store import LineageStore


def test_lineage_save_and_load(tmp_path) -> None:
    store = LineageStore(output_dir=tmp_path)
    saved = store.save_creature(
        None,
        {
            "stage": "basic",
            "name": "Leafbud",
            "types": ["grass"],
            "stats": {"hp": 45, "attack": 40, "defense": 55, "special_attack": 50, "special_defense": 50, "speed": 45},
            "core_motifs": ["leaf shell"],
            "color_palette": ["green"],
            "prompt": "original grass creature",
            "negative_prompt": "text",
            "seed": 123,
            "image_path": "outputs/generations/test.png",
        },
    )
    loaded = store.load(saved["lineage_id"])
    assert loaded["creatures"][0]["name"] == "Leafbud"

