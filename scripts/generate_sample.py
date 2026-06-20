from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evolution.lineage_store import LineageStore
from src.generation.prompt_builder import build_sdxl_prompt
from src.generation.sdxl_pipeline import SDXLGenerator
from src.llm.planner import plan_creature
from src.llm.schemas import CreatureInput, CreatureStats
from src.utils.config import load_environment, load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type-1", default="fire")
    parser.add_argument("--type-2", default="flying")
    parser.add_argument("--appearance", default="a small dragon-like creature with feathered wings and a glowing flame crest")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    load_environment()
    config = load_yaml_config("configs/app.yaml")
    gen_cfg = config.get("generation", {})
    creature_input = CreatureInput(
        type_1=args.type_1,
        type_2=None if args.type_2 == "none" else args.type_2,
        stats=CreatureStats(hp=70, attack=95, defense=70, special_attack=110, special_defense=75, speed=120),
        appearance_description=args.appearance,
    )
    planned = plan_creature(creature_input)
    if planned.plan is None:
        raise SystemExit(planned.error or "Planning failed")
    prompt = build_sdxl_prompt(
        types=creature_input.types,
        stats=creature_input.stats.as_dict(),
        appearance_description=creature_input.appearance_description,
        llm_prompt=planned.plan.sdxl_prompt,
        color_palette=planned.plan.color_palette,
        core_motifs=planned.plan.core_motifs,
        use_lora=args.use_lora,
    )
    if args.metadata_only:
        print(json.dumps({"provider": planned.provider_used, "prompt": prompt, "plan": planned.plan.model_dump()}, indent=2))
        return

    generator = SDXLGenerator(
        model_id=gen_cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
        width=int(gen_cfg.get("width", 768)),
        height=int(gen_cfg.get("height", 768)),
    )
    result = generator.generate(
        prompt=prompt,
        negative_prompt=planned.plan.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.steps or int(gen_cfg.get("num_inference_steps", 20)),
        guidance_scale=float(gen_cfg.get("guidance_scale", 7.0)),
        lora_path=args.lora_path if args.use_lora else None,
    )
    creature = {
        "stage": planned.plan.evolution_stage,
        "name": planned.plan.name,
        "types": planned.plan.types,
        "stats": creature_input.stats.as_dict(),
        "core_motifs": planned.plan.core_motifs,
        "color_palette": planned.plan.color_palette,
        "prompt": prompt,
        "negative_prompt": planned.plan.negative_prompt,
        "seed": result.seed,
        "image_path": result.image_path,
    }
    saved = LineageStore().save_creature(None, creature)
    print(json.dumps({"image": result.image_path, "lineage_id": saved["lineage_id"]}, indent=2))


if __name__ == "__main__":
    main()
