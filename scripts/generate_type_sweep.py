from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from src.generation.prompt_builder import build_negative_prompt, build_sdxl_prompt
from src.generation.sdxl_pipeline import SDXLGenerator
from src.llm.schemas import POKEMON_TYPES
from src.utils.config import PROJECT_ROOT, ensure_dir, load_environment, load_yaml_config


APPEARANCE_BY_TYPE = {
    "normal": "round tan rabbit-like creature with soft ears, simple markings, and a friendly alert pose",
    "fire": "small red salamander creature with a flaming crest, ember spots, and warm glowing tail",
    "water": "blue turtle creature with smooth fins, shell plates, and water droplet markings",
    "grass": "green seed creature with leaf fronds, vine arms, and flower buds on its back",
    "electric": "yellow mouse-like creature with visible lightning bolts, electric arcs, and bright cheek sparks",
    "ice": "white fox creature with ice crystal spikes, frost mist, and snowflake markings",
    "fighting": "compact red martial creature with strong limbs, band-like markings, and a ready stance",
    "poison": "purple lizard creature with toxic spots, vapor sacs, and thorny glands",
    "ground": "brown mole creature with digging claws, rugged hide, and dust cloud markings",
    "flying": "sky-blue bird creature with wide feathered wings, aerodynamic body, and sharp talons",
    "psychic": "pink mystic creature with a glowing forehead gem, floating ornaments, and calm aura",
    "bug": "lime beetle creature with antennae, shell segments, and transparent wings",
    "rock": "gray armadillo creature with stone plates, craggy horns, and mineral shell",
    "ghost": "indigo mask-faced creature with a wispy tail, spectral glow, and floating posture",
    "dragon": "deep blue dragon creature with horns, scales, serpentine tail, and small powerful wings",
    "dark": "black fox creature with sharp mask markings, crescent claws, and stealthy posture",
    "steel": "silver crab creature with metal plates, rivets, and blade-like claw edges",
    "fairy": "pink moth creature with soft wings, star motifs, and sparkling pearl aura",
}


def _stats_for_type(index: int) -> dict[str, int]:
    profiles = [
        {"hp": 70, "attack": 70, "defense": 102, "special_attack": 70, "special_defense": 141, "speed": 32},
        {"hp": 55, "attack": 95, "defense": 55, "special_attack": 60, "special_defense": 60, "speed": 105},
        {"hp": 80, "attack": 60, "defense": 85, "special_attack": 105, "special_defense": 80, "speed": 70},
    ]
    return profiles[index % len(profiles)]


def _contact_sheet(items: list[dict[str, str]], output_path: Path) -> None:
    tile = 256
    label_h = 34
    cols = 6
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, item in enumerate(items):
        image = Image.open(PROJECT_ROOT / item["image_path"]).convert("RGB")
        image.thumbnail((tile, tile))
        x = (idx % cols) * tile
        y = (idx // cols) * (tile + label_h)
        sheet.paste(image, (x + (tile - image.width) // 2, y))
        draw.text((x + 8, y + tile + 8), item["type"], fill=(0, 0, 0))
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=8800)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--output-dir", default="outputs/demo/type_prompt_sweep")
    args = parser.parse_args()

    load_environment()
    config = load_yaml_config("configs/app.yaml")
    gen_cfg = config.get("generation", {})
    output_dir = ensure_dir(args.output_dir)
    lora_path = args.lora_path or gen_cfg.get("lora_path")
    lora_scale = args.lora_scale if args.lora_scale is not None else float(gen_cfg.get("lora_scale", 0.5))
    generator = SDXLGenerator(
        model_id=gen_cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
        output_dir=output_dir,
        width=int(gen_cfg.get("width", 768)),
        height=int(gen_cfg.get("height", 768)),
        torch_dtype=gen_cfg.get("torch_dtype", "fp16"),
    )
    negative_prompt = build_negative_prompt()

    items: list[dict[str, str]] = []
    for index, type_name in enumerate(POKEMON_TYPES):
        stats = _stats_for_type(index)
        appearance = APPEARANCE_BY_TYPE[type_name]
        prompt = build_sdxl_prompt(
            types=[type_name],
            stats=stats,
            appearance_description=appearance,
            use_lora=True,
        )
        result = generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=args.seed + index,
            num_inference_steps=args.steps,
            guidance_scale=float(gen_cfg.get("guidance_scale", 7.0)),
            lora_path=lora_path,
            lora_scale=lora_scale,
        )
        items.append(
            {
                "type": type_name,
                "appearance": appearance,
                "stats": stats,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": str(args.seed + index),
                "image_path": result.image_path,
                "lora_status": result.lora_status,
            }
        )

    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"{stem}_summary.json"
    sheet_path = output_dir / f"{stem}_contact_sheet.png"
    summary_path.write_text(json.dumps({"items": items}, indent=2), encoding="utf-8")
    _contact_sheet(items, sheet_path)
    print(json.dumps({"summary": str(summary_path), "contact_sheet": str(sheet_path), "count": len(items)}, indent=2))


if __name__ == "__main__":
    main()
