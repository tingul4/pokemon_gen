from __future__ import annotations

import json
import random
from typing import Any

import streamlit as st

from src.evolution.evolution_planner import STAGE_DOWN, STAGE_UP, plan_devolution, plan_evolution
from src.evolution.lineage_store import LineageStore
from src.evolution.stage_estimator import estimate_stage
from src.generation.prompt_builder import build_negative_prompt, build_sdxl_prompt
from src.generation.sdxl_pipeline import SDXLGenerator
from src.llm.planner import plan_creature
from src.llm.schemas import CreatureInput, CreatureStats
from src.ui.components import render_creature, stat_sliders, type_selectors
from src.ui.state import cache_creature_stage, default_session_state, get_cached_stage, reset_lineage_state, stage_label
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


def _show_stage_cache() -> None:
    cache = st.session_state.get("stage_cache", {})
    cols = st.columns(3)
    for idx, stage in enumerate(("basic", "stage_1", "stage_2")):
        with cols[idx]:
            status = "已生成" if stage in cache else "尚未生成"
            st.caption(f"{stage_label(stage)}：{status}")


def _switch_to_cached_stage(stage: str) -> bool:
    cached = get_cached_stage(st.session_state, stage)
    if not cached:
        return False
    st.session_state["current_creature"] = cached
    st.session_state["lineage_id"] = cached.get("lineage_id", st.session_state.get("lineage_id"))
    return True


def _run_generation(
    creature_input: CreatureInput,
    settings: dict[str, Any],
    parent_id: str | None = None,
    ip_adapter_image_path: str | None = None,
) -> dict[str, Any]:
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
    negative_prompt = build_negative_prompt(plan.negative_prompt)
    generator = _generator(settings)
    image_result = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=settings["seed"],
        num_inference_steps=settings["steps"],
        guidance_scale=settings["guidance"],
        lora_path=settings["lora_path"] if use_lora else None,
        lora_scale=settings["lora_scale"],
        ip_adapter_enabled=bool(settings["use_ip_adapter"] and ip_adapter_image_path),
        ip_adapter_image_path=ip_adapter_image_path,
        ip_adapter_model_id=settings["ip_adapter_model_id"],
        ip_adapter_subfolder=settings["ip_adapter_subfolder"],
        ip_adapter_weight_name=settings["ip_adapter_weight_name"],
        ip_adapter_scale=settings["ip_adapter_scale"],
    )
    creature = {
        "parent_id": parent_id,
        "stage": creature_input.requested_stage or plan.evolution_stage or stage_estimate["stage"],
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
        "lora_scale": settings["lora_scale"] if use_lora else None,
        "ip_adapter_used": image_result.ip_adapter_used,
        "ip_adapter_status": image_result.ip_adapter_status,
        "ip_adapter_scale": settings["ip_adapter_scale"] if image_result.ip_adapter_used else None,
        "ip_adapter_reference_image": ip_adapter_image_path if image_result.ip_adapter_used else None,
    }
    store = LineageStore()
    saved = store.save_creature(st.session_state.get("lineage_id"), creature)
    st.session_state["lineage_id"] = saved["lineage_id"]
    st.session_state["current_creature"] = saved
    cache_creature_stage(st.session_state, saved)
    return saved


def main() -> None:
    st.set_page_config(page_title="Pokemon-style image generator", layout="wide")
    _init_state()

    generation_cfg = CONFIG.get("generation", {})
    ui_cfg = CONFIG.get("ui", {})
    deployment_mode = bool(ui_cfg.get("deployment_mode", True))
    debug_enabled = bool(ui_cfg.get("show_debug_panel", False))
    with st.sidebar:
        st.header("生成設定")
        st.caption("部署模式：已隱藏模型與本地路徑")
        width = st.select_slider("圖片寬度", options=[512, 768, 1024], value=int(generation_cfg.get("width", 768)))
        height = st.select_slider("圖片高度", options=[512, 768, 1024], value=int(generation_cfg.get("height", 768)))
        seed = st.number_input("隨機種子", min_value=0, max_value=2**31 - 1, value=random.randint(1, 999999), step=1)
        steps = st.slider("推論步數", 1, 50, int(generation_cfg.get("num_inference_steps", 20)))
        guidance = st.slider("提示詞引導強度", 1.0, 15.0, float(generation_cfg.get("guidance_scale", 7.0)), 0.5)
        ip_adapter_scale = st.slider("進化鏈一致性強度", 0.0, 1.0, float(generation_cfg.get("ip_adapter_scale", 0.45)), 0.05)
        debug_enabled = st.toggle("顯示進階資訊", value=debug_enabled)

        model_id = generation_cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0")
        use_lora = bool(generation_cfg.get("use_lora", False))
        lora_path = generation_cfg.get("lora_path", "")
        lora_scale = float(generation_cfg.get("lora_scale", 0.5))
        ip_adapter_model_id = generation_cfg.get("ip_adapter_model_id", "h94/IP-Adapter")
        ip_adapter_subfolder = generation_cfg.get("ip_adapter_subfolder", "sdxl_models")
        ip_adapter_weight_name = generation_cfg.get("ip_adapter_weight_name", "ip-adapter_sdxl.bin")

        if not deployment_mode and debug_enabled:
            st.divider()
            st.caption("開發模式設定")
            model_id = st.text_input("SDXL 模型", model_id)
            use_lora = st.toggle("使用 LoRA", value=use_lora)
            lora_scale = st.slider("LoRA 強度", 0.0, 1.0, lora_scale, 0.05)
            ip_adapter_model_id = st.text_input("IP-Adapter 模型", ip_adapter_model_id)
            ip_adapter_subfolder = st.text_input("IP-Adapter 子資料夾", ip_adapter_subfolder)
            ip_adapter_weight_name = st.text_input("IP-Adapter 權重檔名", ip_adapter_weight_name)

    settings = {
        "model_id": model_id,
        "width": int(width),
        "height": int(height),
        "use_lora": use_lora,
        "lora_path": lora_path,
        "lora_scale": float(lora_scale),
        "use_ip_adapter": True,
        "ip_adapter_model_id": ip_adapter_model_id,
        "ip_adapter_subfolder": ip_adapter_subfolder,
        "ip_adapter_weight_name": ip_adapter_weight_name,
        "ip_adapter_scale": float(ip_adapter_scale),
        "seed": int(seed),
        "steps": int(steps),
        "guidance": float(guidance),
    }

    st.title("Pokemon-style image generator")
    st.caption("學術展示用途，只生成原創怪獸；本專案與 Nintendo、Game Freak、Creatures Inc. 或 The Pokemon Company 無關。")

    with st.form("creature_form"):
        type_1, type_2 = type_selectors()
        stats = stat_sliders()
        appearance = st.text_area(
            "外觀描述",
            "小型龍形怪獸，有羽毛翅膀和發光火焰冠飾",
            height=120,
        )
        submitted = st.form_submit_button("生成原創怪獸")

    if submitted:
        try:
            reset_lineage_state(st.session_state)
            creature_input = _build_input(type_1, type_2, stats, appearance)
            with st.spinner("正在規劃設定並生成圖片..."):
                _run_generation(creature_input, settings)
        except Exception as exc:
            st.error(f"生成失敗：{exc}")

    current = st.session_state.get("current_creature")
    if current:
        _show_stage_cache()
        left, right = st.columns([2, 1])
        with left:
            render_creature(current)
        with right:
            st.subheader("進化鏈")
            can_evolve = current.get("stage") in {"basic", "stage_1"}
            can_devolve = current.get("stage") in {"stage_1", "stage_2"}
            if st.button("進化", disabled=not can_evolve):
                try:
                    target_stage = STAGE_UP.get(current.get("stage"))
                    if target_stage and _switch_to_cached_stage(target_stage):
                        st.rerun()
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
                    with st.spinner("正在生成進化型..."):
                        _run_generation(
                            creature_input,
                            settings,
                            parent_id=current.get("creature_id"),
                            ip_adapter_image_path=current.get("image_path"),
                        )
                    st.rerun()
                except Exception as exc:
                    st.error(f"進化失敗：{exc}")
            if st.button("退化", disabled=not can_devolve):
                try:
                    target_stage = STAGE_DOWN.get(current.get("stage"))
                    if target_stage and _switch_to_cached_stage(target_stage):
                        st.rerun()
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
                    with st.spinner("正在生成退化型..."):
                        _run_generation(
                            creature_input,
                            settings,
                            parent_id=current.get("creature_id"),
                            ip_adapter_image_path=current.get("image_path"),
                        )
                    st.rerun()
                except Exception as exc:
                    st.error(f"退化失敗：{exc}")

        if debug_enabled:
            with st.expander("進階資訊"):
                st.write(f"LLM 供應商：`{current.get('llm_provider')}`")
                st.write(f"LoRA 狀態：`{'已啟用' if current.get('lora_used') else '未啟用'}`")
                st.write(f"IP-Adapter 狀態：`{'已啟用' if current.get('ip_adapter_used') else '未啟用'}`")
                st.write(f"隨機種子：`{current.get('seed')}`")
                st.text_area("最終提示詞", current.get("prompt", ""), height=140)
                st.text_area("負向提示詞", current.get("negative_prompt", ""), height=80)
                if current.get("llm_warnings"):
                    st.warning("\n".join(current["llm_warnings"]))


if __name__ == "__main__":
    main()
