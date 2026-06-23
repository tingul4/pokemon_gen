from __future__ import annotations

import csv
import json

from PIL import Image

from src.data.prepare_dataset import find_images, prepare_lora_dataset


FIELDS = [
    "national_number",
    "gen",
    "english_name",
    "japanese_name",
    "primary_type",
    "secondary_type",
    "classification",
    "percent_male",
    "percent_female",
    "height_m",
    "weight_kg",
    "capture_rate",
    "base_egg_steps",
    "hp",
    "attack",
    "defense",
    "sp_attack",
    "sp_defense",
    "speed",
    "abilities_0",
    "abilities_1",
    "abilities_2",
    "abilities_hidden",
    "is_sublegendary",
    "is_legendary",
    "is_mythical",
    "evochain_0",
    "evochain_1",
    "evochain_2",
    "evochain_3",
    "evochain_4",
    "evochain_5",
    "evochain_6",
    "gigantamax",
    "mega_evolution",
    "mega_evolution_alt",
    "description",
]


def _write_pokedex_csv(path, rows) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-16", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_find_images_uses_large_images_only(tmp_path, monkeypatch) -> None:
    import src.data.prepare_dataset as prepare_dataset

    monkeypatch.setattr(prepare_dataset, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "data" / "raw" / "cristobalmitchell_pokedex"
    _write_pokedex_csv(root / "data" / "pokemon.csv", [])
    for subdir, filename in [
        ("large_images", "004.png"),
        ("alt_images", "004Charmander-Mega.png"),
        ("small_images", "004.png"),
        ("type_icons", "fire.png"),
    ]:
        image_dir = root / "images" / subdir
        image_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (255, 80, 20)).save(image_dir / filename)

    images = find_images(root)

    assert [path.parent.name for path in images] == ["large_images"]


def test_prepare_lora_dataset_writes_annotations_from_pokedex_csv(tmp_path, monkeypatch) -> None:
    import src.data.prepare_dataset as prepare_dataset

    monkeypatch.setattr(prepare_dataset, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "data" / "raw" / "cristobalmitchell_pokedex"
    _write_pokedex_csv(
        root / "data" / "pokemon.csv",
        [
            {
                "national_number": "4",
                "gen": "1",
                "english_name": "Charmander",
                "japanese_name": "",
                "primary_type": "fire",
                "secondary_type": "",
                "classification": "Lizard Pokémon",
                "percent_male": "",
                "percent_female": "",
                "height_m": "0.6",
                "weight_kg": "8.5",
                "capture_rate": "",
                "base_egg_steps": "",
                "hp": "39",
                "attack": "52",
                "defense": "43",
                "sp_attack": "60",
                "sp_defense": "50",
                "speed": "65",
                "abilities_0": "Blaze",
                "abilities_1": "",
                "abilities_2": "",
                "abilities_hidden": "Solar Power",
                "is_sublegendary": "0",
                "is_legendary": "0",
                "is_mythical": "0",
                "evochain_0": "Charmander",
                "evochain_1": "Level",
                "evochain_2": "Charmeleon",
                "evochain_3": "",
                "evochain_4": "",
                "evochain_5": "",
                "evochain_6": "",
                "gigantamax": "",
                "mega_evolution": "",
                "mega_evolution_alt": "",
                "description": "It has a preference for hot things. When it rains, steam is said to spout from the tip of its tail.",
            }
        ],
    )
    image_dir = root / "images" / "large_images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), (255, 80, 20)).save(image_dir / "004.png")

    rows = prepare_lora_dataset(
        raw_image_dir=root,
        output_image_dir=tmp_path / "data" / "processed" / "lora_images",
        captions_path=tmp_path / "data" / "processed" / "captions.jsonl",
        annotations_path=tmp_path / "data" / "processed" / "annotations.jsonl",
        metadata_path=tmp_path / "data" / "processed" / "metadata.json",
        resolution=64,
    )

    assert len(rows) == 1
    annotation = json.loads((tmp_path / "data" / "processed" / "annotations.jsonl").read_text(encoding="utf-8"))
    assert annotation["pokemon_name"] == "Charmander"
    assert annotation["national_number"] == 4
    assert annotation["label"]["types"] == ["fire"]
    assert annotation["label"]["stats"]["speed"] == 65
    assert annotation["label"]["appearance_description"].startswith("Lizard creature")
    assert "height 0.6 m" not in annotation["label"]["appearance_description"]
    assert annotation["caption"].startswith(
        "sks style, single image, single creature, full body, blank background, clean composition"
    )
    assert "types fire, stats hp39 attack52 defense43" in annotation["caption"]
    assert "special_attack60 special_defense50 speed65" in annotation["caption"]
    assert "appearance Lizard creature" in annotation["caption"]
    assert "hot things" in annotation["caption"]
    assert "ability" not in annotation["caption"]
    assert rows[0]["image"].endswith("data/processed/lora_images/004.png")
    assert "\\u" not in (tmp_path / "data" / "processed" / "annotations.jsonl").read_text(encoding="utf-8")
