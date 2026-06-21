from __future__ import annotations

import json

from PIL import Image

from src.data.prepare_dataset import dataset_name_to_pokeapi_name, prepare_lora_dataset


def test_dataset_name_to_pokeapi_name_handles_special_cases() -> None:
    assert dataset_name_to_pokeapi_name("Mr. Mime") == "mr-mime"
    assert dataset_name_to_pokeapi_name("Porygon-Z") == "porygon-z"
    assert dataset_name_to_pokeapi_name("Tapu Koko") == "tapu-koko"
    assert dataset_name_to_pokeapi_name("NidoranΓÖÇ") == "nidoran-f"
    assert dataset_name_to_pokeapi_name("NidoranΓÖé") == "nidoran-m"


def test_prepare_lora_dataset_writes_structured_annotations(tmp_path, monkeypatch) -> None:
    import src.data.prepare_dataset as prepare_dataset

    monkeypatch.setattr(prepare_dataset, "PROJECT_ROOT", tmp_path)
    raw_dir = tmp_path / "data" / "raw" / "images" / "Charmander"
    raw_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), (255, 80, 20)).save(raw_dir / "0.jpg")

    metadata_dir = tmp_path / "data" / "processed"
    metadata_dir.mkdir(parents=True)
    metadata = [
        {
            "id": 4,
            "name": "charmander",
            "types": ["fire"],
            "stats": {
                "hp": 39,
                "attack": 52,
                "defense": 43,
                "special_attack": 60,
                "special_defense": 50,
                "speed": 65,
            },
            "base_stat_total": 309,
            "height": 6,
            "weight": 85,
            "abilities": ["blaze"],
            "species_profile": {
                "genus": "Lizard Pokemon",
                "official_flavor_text": "The flame on its tail shows its life force.",
                "flavor_version": "sword",
                "color": "red",
                "shape": "upright",
                "habitat": "mountain",
            },
        }
    ]
    (metadata_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    rows = prepare_lora_dataset(
        raw_image_dir=tmp_path / "data" / "raw" / "images",
        output_image_dir=tmp_path / "data" / "processed" / "lora_images",
        captions_path=tmp_path / "data" / "processed" / "captions.jsonl",
        annotations_path=tmp_path / "data" / "processed" / "annotations.jsonl",
        metadata_path=tmp_path / "data" / "processed" / "metadata.json",
        resolution=64,
    )

    assert len(rows) == 1
    annotation = json.loads((tmp_path / "data" / "processed" / "annotations.jsonl").read_text(encoding="utf-8"))
    assert annotation["pokeapi_name"] == "charmander"
    assert annotation["label"]["types"] == ["fire"]
    assert annotation["label"]["stats"]["speed"] == 65
    assert annotation["label"]["species_profile"]["genus"] == "Lizard Pokemon"
    assert "official Pokedex note (sword)" in annotation["label"]["appearance_description"]
    assert "color: red" in annotation["label"]["appearance_description"]
    assert "appearance_description" in annotation["label"]
    assert "stats hp39 atk52 def43 spa60 spd50 spe65" in annotation["caption"]
    assert "lizard creature" in annotation["caption"]
    assert "The flame on its tail" not in annotation["caption"]
