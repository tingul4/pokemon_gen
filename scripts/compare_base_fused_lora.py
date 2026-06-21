from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw

from src.generation.prompt_builder import build_sdxl_prompt
from src.llm.planner import plan_creature
from src.llm.schemas import CreatureInput, CreatureStats
from src.utils.config import PROJECT_ROOT, ensure_dir, load_environment, load_yaml_config


def _load_pipe(model_id: str, torch_dtype: str) -> StableDiffusionXLPipeline:
    dtype = torch.float16 if torch_dtype == "fp16" and torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype is torch.float16 else None,
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    return pipe


def _generate(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
) -> Image.Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]


def _contact_sheet(base_image: Image.Image, fused_image: Image.Image) -> Image.Image:
    label_h = 36
    sheet = Image.new("RGB", (base_image.width * 2, base_image.height + label_h), "white")
    sheet.paste(base_image, (0, label_h))
    sheet.paste(fused_image, (base_image.width, label_h))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "Base SDXL", fill=(0, 0, 0))
    draw.text((base_image.width + 12, 10), "Fused SDXL + LoRA", fill=(0, 0, 0))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed-condition base SDXL vs fused LoRA comparison.")
    parser.add_argument("--type-1", default="fire")
    parser.add_argument("--type-2", default="flying")
    parser.add_argument("--appearance", default="a small dragon-like creature with feathered wings and a glowing flame crest")
    parser.add_argument("--hp", type=int, default=70)
    parser.add_argument("--attack", type=int, default=95)
    parser.add_argument("--defense", type=int, default=70)
    parser.add_argument("--special-attack", type=int, default=110)
    parser.add_argument("--special-defense", type=int, default=75)
    parser.add_argument("--speed", type=int, default=120)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--output-dir", default="outputs/demo/base_vs_fused_lora")
    parser.add_argument("--metadata-file", default=None, help="Reuse prompt and condition from a previous comparison metadata JSON.")
    args = parser.parse_args()

    load_environment()
    config = load_yaml_config("configs/app.yaml")
    gen_cfg = config.get("generation", {})
    output_dir = ensure_dir(args.output_dir)

    if args.metadata_file:
        previous = json.loads(Path(args.metadata_file).read_text(encoding="utf-8"))
        condition = previous["condition"]
        prompt = previous["prompt"]
        negative_prompt = previous["negative_prompt"]
        provider = previous.get("provider")
        warnings = previous.get("warnings", [])
        plan = previous.get("plan", {})
    else:
        creature_input = CreatureInput(
            type_1=args.type_1,
            type_2=None if args.type_2 == "none" else args.type_2,
            stats=CreatureStats(
                hp=args.hp,
                attack=args.attack,
                defense=args.defense,
                special_attack=args.special_attack,
                special_defense=args.special_defense,
                speed=args.speed,
            ),
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
            use_lora=False,
        )
        negative_prompt = planned.plan.negative_prompt
        provider = planned.provider_used
        warnings = planned.warnings
        plan = planned.plan.model_dump()
        condition = {
            "types": creature_input.types,
            "stats": creature_input.stats.as_dict(),
            "appearance": creature_input.appearance_description,
        }
    steps = args.steps or int(gen_cfg.get("num_inference_steps", 20))
    width = int(gen_cfg.get("width", 768))
    height = int(gen_cfg.get("height", 768))
    guidance_scale = float(gen_cfg.get("guidance_scale", 7.0))
    model_id = gen_cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0")
    torch_dtype = gen_cfg.get("torch_dtype", "fp16")

    base_pipe = _load_pipe(model_id, torch_dtype)
    base_image = _generate(base_pipe, prompt, negative_prompt, args.seed, steps, guidance_scale, width, height)
    del base_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fused_pipe = _load_pipe(model_id, torch_dtype)
    fused_pipe.load_lora_weights(args.lora_path)
    fused_pipe.fuse_lora(lora_scale=args.lora_scale)
    fused_image = _generate(fused_pipe, prompt, negative_prompt, args.seed, steps, guidance_scale, width, height)
    del fused_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stem = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    base_path = output_dir / f"{stem}_base_sdxl.png"
    fused_path = output_dir / f"{stem}_fused_lora.png"
    sheet_path = output_dir / f"{stem}_comparison.png"
    metadata_path = output_dir / f"{stem}_metadata.json"
    base_image.save(base_path)
    fused_image.save(fused_path)
    _contact_sheet(base_image, fused_image).save(sheet_path)

    metadata = {
        "condition": condition,
        "provider": provider,
        "warnings": warnings,
        "plan": plan,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": args.seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "model_id": model_id,
        "lora_path": args.lora_path,
        "lora_scale": args.lora_scale,
        "outputs": {
            "base_sdxl": str(base_path.relative_to(PROJECT_ROOT)),
            "fused_lora": str(fused_path.relative_to(PROJECT_ROOT)),
            "comparison": str(sheet_path.relative_to(PROJECT_ROOT)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": metadata["outputs"], "metadata": str(metadata_path.relative_to(PROJECT_ROOT))}, indent=2))


if __name__ == "__main__":
    main()
