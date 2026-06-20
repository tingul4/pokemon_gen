from __future__ import annotations

import json
import re
from typing import Callable

from pydantic import ValidationError

from src.evolution.stage_estimator import estimate_stage
from src.generation.prompt_builder import build_negative_prompt, type_hints
from src.llm.gemini_client import GeminiPlannerClient
from src.llm.groq_client import GroqPlannerClient
from src.llm.schemas import CreatureInput, CreaturePlan, PlannedCreatureResult
from src.utils.config import load_environment, load_yaml_config


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def build_planner_prompt(creature_input: CreatureInput, stage: str) -> str:
    payload = {
        "types": creature_input.types,
        "stats": creature_input.stats.as_dict(),
        "appearance_description": creature_input.appearance_description,
        "evolution_stage": creature_input.requested_stage or stage,
        "lineage_constraints": creature_input.lineage_constraints or "",
    }
    return (
        "Create a structured plan for an original Pokemon-style creature for an academic demo. "
        "Do not create or name an official Pokemon character and do not claim affiliation with Nintendo, "
        "Game Freak, Creatures Inc., or The Pokemon Company. Preserve the requested types and stats. "
        "Convert stats into visible creature traits. Return JSON only with keys: name, types, "
        "evolution_stage, visual_concept, stat_interpretation, core_motifs, color_palette, "
        "sdxl_prompt, negative_prompt, pokedex_entry, evolution_hint, devolution_hint.\n\n"
        f"Input JSON:\n{json.dumps(payload, indent=2)}"
    )


def repair_prompt(original_prompt: str, bad_response: str) -> str:
    return (
        f"{original_prompt}\n\nThe previous response was invalid JSON. Repair it and return one valid JSON object only. "
        f"Invalid response:\n{bad_response}"
    )


def deterministic_plan(creature_input: CreatureInput, stage: str) -> CreaturePlan:
    colors, motifs = type_hints(creature_input.types)
    if not colors:
        colors = ["bright", "contrasting", "clean"]
    if not motifs:
        motifs = ["expressive eyes", "rounded monster silhouette"]
    stats = creature_input.stats.as_dict()
    stat_text = {
        name: f"{name.replace('_', ' ').title()} value {value} influences the creature silhouette."
        for name, value in stats.items()
    }
    type_text = " and ".join(creature_input.types)
    visual_concept = (
        f"An original {type_text} creature with {', '.join(motifs[:3])}, "
        f"using {', '.join(colors[:3])} colors. {creature_input.appearance_description}"
    )
    return CreaturePlan(
        name=f"{creature_input.type_1.title()}mote",
        types=creature_input.types,
        evolution_stage=creature_input.requested_stage or stage,
        visual_concept=visual_concept,
        stat_interpretation=stat_text,
        core_motifs=motifs[:5],
        color_palette=colors[:5],
        sdxl_prompt=visual_concept,
        negative_prompt=build_negative_prompt(),
        pokedex_entry=(
            f"This original {type_text} creature channels its elemental traits through a distinctive, "
            "playful silhouette while avoiding contact with humans until it feels safe."
        ),
        evolution_hint="The evolved form should grow larger and gain stronger elemental ornaments.",
        devolution_hint="The previous form should be smaller, softer, and less ornamented.",
    )


class FallbackLLMPlanner:
    def __init__(
        self,
        gemini_factory: Callable[[], GeminiPlannerClient] | None = None,
        groq_factory: Callable[[], GroqPlannerClient] | None = None,
    ) -> None:
        load_environment()
        config = load_yaml_config("configs/app.yaml")
        gemini_model = config.get("llm", {}).get("gemini_model", "gemini-2.5-flash")
        self.gemini_factory = gemini_factory or (lambda: GeminiPlannerClient(gemini_model))
        self.groq_factory = groq_factory or (lambda: GroqPlannerClient())

    def _try_provider(self, provider: str, client: object, prompt: str) -> tuple[CreaturePlan, str]:
        raw = client.generate(prompt)  # type: ignore[attr-defined]
        try:
            return CreaturePlan.model_validate(_extract_json(raw)), raw
        except (json.JSONDecodeError, ValidationError) as exc:
            repaired_raw = client.generate(repair_prompt(prompt, raw))  # type: ignore[attr-defined]
            try:
                return CreaturePlan.model_validate(_extract_json(repaired_raw)), repaired_raw
            except Exception as repair_exc:
                raise RuntimeError(f"{provider} returned invalid JSON: {exc}; repair failed: {repair_exc}") from repair_exc

    def plan(self, creature_input: CreatureInput) -> PlannedCreatureResult:
        stage = estimate_stage(creature_input.stats.as_dict())["stage"]
        target_stage = creature_input.requested_stage or stage
        prompt = build_planner_prompt(creature_input, stage)
        errors: list[str] = []
        for provider, factory in (("gemini", self.gemini_factory), ("groq", self.groq_factory)):
            try:
                plan, raw = self._try_provider(provider, factory(), prompt)
                plan = plan.model_copy(update={"types": creature_input.types, "evolution_stage": target_stage})
                return PlannedCreatureResult(ok=True, plan=plan, provider_used=provider, raw_response=raw, warnings=errors)
            except Exception as exc:
                errors.append(f"{provider}: {exc}")

        fallback = deterministic_plan(creature_input, stage)
        return PlannedCreatureResult(
            ok=True,
            plan=fallback,
            provider_used="deterministic",
            raw_response=fallback.model_dump(),
            warnings=errors + ["Used deterministic fallback plan because LLM providers failed."],
        )
