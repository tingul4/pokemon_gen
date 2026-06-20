from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import streamlit as st

from src.evolution.evolution_planner import plan_devolution, plan_evolution
from src.evolution.lineage_store import LineageStore
from src.evolution.stage_estimator import estimate_stage
from src.generation.prompt_builder import build_negative_prompt, build_sdxl_prompt
from src.generation.sdxl_pipeline import SDXLGenerator
from src.llm.planner import plan_creature
from src.llm.schemas import CreatureInput, CreatureStats
from src.ui.components import render_creature, stat_sliders, type_selectors
from src.ui.state import default_session_state
from src.utils.config import load_environment, load_yaml_config


load_environment()
CONFIG = load_yaml_config("configs/app.yaml")


def _init_state() -> None:
    for key, value in default_session_state().items():
        st.session_state.setdefault(key, value)


def _generator(settings: dict[str, Any]) -> SDXLGenerator:
    generation_cfg = CONFIG.get("generation", {})
    return SDXLGenerator(
        model_id=settings["model_id"],
        output_dir=generation_cfg.get("output_dir", "outputs/generations"),
        width=settings["width"],
        height=settings["height"],
        torch_dtype=generation_cfg.get("torch_dtype", "fp16"),
    )


def _build_input(type_1: str, type_2: str | None, stats: dict[str, int], appearance: str, stage: str | None = None) -> CreatureInput:
    return CreatureInput(
        type_1=type_1,
        type_2=type_2,
        stats=CreatureStats(**stats),
        appearance_description=appearance,
        requested_stage=stage,
    )


def _run_generation(creature_input: CreatureInput, settings: dict[str, Any], parent_id: str | None = None) -> dict[str, Any]:
    stage_estimate = estimate_stage(creature_input.stats.as_dict())
    planned = plan_creature(creature_input)
    if not planned.ok or planned.plan is None:
        raise RuntimeError(planned.error or "LLM planning failed.")
    plan = planned.plan
    use_lora = bool(settings["use_lora"] and settings["lora_path"])
    prompt = build_sdxl_prompt(
        types=creature_input.types,
        stats=creature_input.stats.as_dict(),
        appearance_description=creature_input.appearance_description,
        llm_prompt=plan.sdxl_prompt,
        color_palette=plan.color_palette,
        core_motifs=plan.core_motifs,
        use_lora=use_lora,
    )
    negative_prompt = plan.negative_prompt or build_negative_prompt()
    generator = _generator(settings)
    image_result = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=settings["seed"],
        num_inference_steps=settings["steps"],
        guidance_scale=settings["guidance"],
        lora_path=settings["lora_path"] if use_lora else None,
    )
    creature = {
        "parent_id": parent_id,
        "stage": plan.evolution_stage or stage_estimate["stage"],
        "name": plan.name,
        "types": plan.types,
        "stats": creature_input.stats.as_dict(),
        "core_motifs": plan.core_motifs,
        "color_palette": plan.color_palette,
        "visual_concept": plan.visual_concept,
        "pokedex_entry": plan.pokedex_entry,
        "evolution_hint": plan.evolution_hint,
        "devolution_hint": plan.devolution_hint,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": image_result.seed,
        "image_path": image_result.image_path,
        "llm_provider": planned.provider_used,
        "raw_llm_json": planned.raw_response,
        "llm_warnings": planned.warnings,
        "lora_used": image_result.lora_used,
        "lora_status": image_result.lora_status,
    }
    store = LineageStore()
    saved = store.save_creature(st.session_state.get("lineage_id"), creature)
    st.session_state["lineage_id"] = saved["lineage_id"]
    st.session_state["current_creature"] = saved
    return saved


def main() -> None:
    st.set_page_config(page_title="Original Creature Generator", layout="wide")
    _init_state()

    generation_cfg = CONFIG.get("generation", {})
    with st.sidebar:
        st.header("Generation")
        st.write("Primary LLM: Gemini 2.5 Flash")
        st.write("Fallback LLM: Groq")
        model_id = st.text_input("SDXL model", generation_cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"))
        width = st.select_slider("Width", options=[512, 768, 1024], value=int(generation_cfg.get("width", 768)))
        height = st.select_slider("Height", options=[512, 768, 1024], value=int(generation_cfg.get("height", 768)))
        use_lora = st.toggle("Use LoRA", value=bool(generation_cfg.get("use_lora", False)))
        lora_path = st.text_input("LoRA path", generation_cfg.get("lora_path", ""))
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=random.randint(1, 999999), step=1)
        steps = st.slider("Inference steps", 1, 50, int(generation_cfg.get("num_inference_steps", 20)))
        guidance = st.slider("Guidance scale", 1.0, 15.0, float(generation_cfg.get("guidance_scale", 7.0)), 0.5)

    settings = {
        "model_id": model_id,
        "width": int(width),
        "height": int(height),
        "use_lora": use_lora,
        "lora_path": lora_path,
        "seed": int(seed),
        "steps": int(steps),
        "guidance": float(guidance),
    }

    st.title("Conditional Pokemon-style Original Creature Generator")
    st.caption("Academic demo. Generates original creatures and does not claim affiliation with Nintendo, Game Freak, Creatures Inc., or The Pokemon Company.")

    with st.form("creature_form"):
        type_1, type_2 = type_selectors()
        stats = stat_sliders()
        appearance = st.text_area(
            "Appearance description",
            "a small dragon-like creature with feathered wings and a glowing flame crest",
            height=120,
        )
        submitted = st.form_submit_button("Generate")

    if submitted:
        try:
            creature_input = _build_input(type_1, type_2, stats, appearance)
            with st.spinner("Planning prompt and generating image..."):
                _run_generation(creature_input, settings)
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    current = st.session_state.get("current_creature")
    if current:
        left, right = st.columns([2, 1])
        with left:
            render_creature(current)
        with right:
            st.subheader("Lineage Controls")
            can_evolve = current.get("stage") in {"basic", "stage_1"}
            can_devolve = current.get("stage") in {"stage_1", "stage_2"}
            if st.button("Evolve", disabled=not can_evolve):
                try:
                    evolved = plan_evolution(current)
                    creature_input = _build_input(
                        evolved["types"][0],
                        evolved["types"][1] if len(evolved.get("types", [])) > 1 else None,
                        evolved["stats"],
                        evolved["appearance_description"],
                        evolved["stage"],
                    )
                    creature_input.lineage_constraints = json.dumps(
                        {"core_motifs": current.get("core_motifs", []), "color_palette": current.get("color_palette", [])}
                    )
                    with st.spinner("Generating evolved form..."):
                        _run_generation(creature_input, settings, parent_id=current.get("creature_id"))
                    st.rerun()
                except Exception as exc:
                    st.error(f"Evolution failed: {exc}")
            if st.button("Devolve", disabled=not can_devolve):
                try:
                    devolved = plan_devolution(current)
                    creature_input = _build_input(
                        devolved["types"][0],
                        devolved["types"][1] if len(devolved.get("types", [])) > 1 else None,
                        devolved["stats"],
                        devolved["appearance_description"],
                        devolved["stage"],
                    )
                    creature_input.lineage_constraints = json.dumps(
                        {"core_motifs": current.get("core_motifs", []), "color_palette": current.get("color_palette", [])}
                    )
                    with st.spinner("Generating previous form..."):
                        _run_generation(creature_input, settings, parent_id=current.get("creature_id"))
                    st.rerun()
                except Exception as exc:
                    st.error(f"Devolution failed: {exc}")

        with st.expander("Debug"):
            st.write(f"LLM provider used: `{current.get('llm_provider')}`")
            st.write(f"LoRA: `{current.get('lora_status')}`")
            st.write(f"Seed: `{current.get('seed')}`")
            st.write(f"Image path: `{current.get('image_path')}`")
            st.text_area("Final SDXL prompt", current.get("prompt", ""), height=140)
            st.text_area("Negative prompt", current.get("negative_prompt", ""), height=80)
            st.json(current.get("raw_llm_json", {}))
            if current.get("llm_warnings"):
                st.warning("\n".join(current["llm_warnings"]))


if __name__ == "__main__":
    main()

