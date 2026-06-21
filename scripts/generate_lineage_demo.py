from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evolution.evolution_planner import plan_devolution, plan_evolution
from src.evolution.lineage_store import LineageStore
from src.generation.prompt_builder import build_sdxl_prompt
from src.generation.sdxl_pipeline import SDXLGenerator
from src.llm.planner import plan_creature
from src.llm.schemas import CreatureInput, CreatureStats
from src.utils.config import load_environment, load_yaml_config


def _input(type_1: str, type_2: str | None, stats: dict[str, int], appearance: str, stage: str | None = None) -> CreatureInput:
    return CreatureInput(
        type_1=type_1,
        type_2=type_2,
        stats=CreatureStats(**stats),
        appearance_description=appearance,
        requested_stage=stage,
    )


def _generate(
    *,
    creature_input: CreatureInput,
    generator: SDXLGenerator,
    store: LineageStore,
    lineage_id: str | None,
    parent_id: str | None,
    seed: int,
    steps: int,
    guidance: float,
    lora_path: str | None,
) -> dict[str, Any]:
    planned = plan_creature(creature_input)
    if planned.plan is None:
        raise RuntimeError(planned.error or "Planning failed.")
    plan = planned.plan
    prompt = build_sdxl_prompt(
        types=creature_input.types,
        stats=creature_input.stats.as_dict(),
        appearance_description=creature_input.appearance_description,
        llm_prompt=plan.sdxl_prompt,
        color_palette=plan.color_palette,
        core_motifs=plan.core_motifs,
        use_lora=bool(lora_path),
    )
    result = generator.generate(
        prompt=prompt,
        negative_prompt=plan.negative_prompt,
        seed=seed,
        num_inference_steps=steps,
        guidance_scale=guidance,
        lora_path=lora_path,
    )
    creature = {
        "parent_id": parent_id,
        "stage": plan.evolution_stage,
        "name": plan.name,
        "types": creature_input.types,
        "stats": creature_input.stats.as_dict(),
        "core_motifs": plan.core_motifs,
        "color_palette": plan.color_palette,
        "visual_concept": plan.visual_concept,
        "pokedex_entry": plan.pokedex_entry,
        "prompt": prompt,
        "negative_prompt": plan.negative_prompt,
        "seed": result.seed,
        "image_path": result.image_path,
        "llm_provider": planned.provider_used,
        "lora_used": result.lora_used,
        "lora_status": result.lora_status,
    }
    return store.save_creature(lineage_id, creature)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-path", default="outputs/lora/pokecreature_sdxl_lora_pokedex")
    parser.add_argument("--output-dir", default="outputs/demo/lineage_tests")
    args = parser.parse_args()

    load_environment()
    config = load_yaml_config("configs/app.yaml")
    gen_cfg = config.get("generation", {})
    lora_path = args.lora_path if args.use_lora else None
    generator = SDXLGenerator(
        model_id=gen_cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
        output_dir=args.output_dir,
        width=int(gen_cfg.get("width", 768)),
        height=int(gen_cfg.get("height", 768)),
    )
    store = LineageStore(output_dir=Path(args.output_dir) / "lineages")

    basic_input = _input(
        "grass",
        None,
        {"hp": 45, "attack": 45, "defense": 55, "special_attack": 45, "special_defense": 55, "speed": 45},
        "small turtle-like creature with a leafy shell and sprout tail",
    )
    basic = _generate(
        creature_input=basic_input,
        generator=generator,
        store=store,
        lineage_id=None,
        parent_id=None,
        seed=args.seed,
        steps=args.steps,
        guidance=float(gen_cfg.get("guidance_scale", 7.0)),
        lora_path=lora_path,
    )
    evolved_plan = plan_evolution(basic)
    evolved_input = _input(
        evolved_plan["types"][0],
        evolved_plan["types"][1] if len(evolved_plan.get("types", [])) > 1 else None,
        evolved_plan["stats"],
        evolved_plan["appearance_description"],
        evolved_plan["stage"],
    )
    evolved = _generate(
        creature_input=evolved_input,
        generator=generator,
        store=store,
        lineage_id=basic["lineage_id"],
        parent_id=basic["creature_id"],
        seed=args.seed + 1,
        steps=args.steps,
        guidance=float(gen_cfg.get("guidance_scale", 7.0)),
        lora_path=lora_path,
    )

    stage_two_input = _input(
        "dragon",
        "electric",
        {"hp": 95, "attack": 125, "defense": 95, "special_attack": 120, "special_defense": 90, "speed": 100},
        "fast thunder dragon with crystal horns and armored scales",
    )
    stage_two = _generate(
        creature_input=stage_two_input,
        generator=generator,
        store=store,
        lineage_id=None,
        parent_id=None,
        seed=args.seed + 2,
        steps=args.steps,
        guidance=float(gen_cfg.get("guidance_scale", 7.0)),
        lora_path=lora_path,
    )
    devolved_plan = plan_devolution(stage_two)
    devolved_input = _input(
        devolved_plan["types"][0],
        devolved_plan["types"][1] if len(devolved_plan.get("types", [])) > 1 else None,
        devolved_plan["stats"],
        devolved_plan["appearance_description"],
        devolved_plan["stage"],
    )
    devolved = _generate(
        creature_input=devolved_input,
        generator=generator,
        store=store,
        lineage_id=stage_two["lineage_id"],
        parent_id=stage_two["creature_id"],
        seed=args.seed + 3,
        steps=args.steps,
        guidance=float(gen_cfg.get("guidance_scale", 7.0)),
        lora_path=lora_path,
    )

    summary = {"basic": basic, "evolved": evolved, "stage_two": stage_two, "devolved": devolved}
    summary_path = Path(args.output_dir) / "lineage_demo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "images": [item["image_path"] for item in summary.values()]}, indent=2))


if __name__ == "__main__":
    main()
