from __future__ import annotations

from src.llm.client import FallbackLLMPlanner
from src.llm.schemas import CreatureInput, CreaturePlan, CreatureStats


VALID_PLAN = {
    "name": "Emberwing",
    "types": ["fire", "flying"],
    "evolution_stage": "stage_2",
    "visual_concept": "Original fiery winged creature.",
    "stat_interpretation": {
        "hp": "balanced",
        "attack": "sharp claws",
        "defense": "light armor",
        "special_attack": "flame aura",
        "special_defense": "warm shield",
        "speed": "aerodynamic",
    },
    "core_motifs": ["flame crest", "feathered wings"],
    "color_palette": ["orange", "crimson"],
    "sdxl_prompt": "original fire flying creature",
    "negative_prompt": "official pokemon",
    "pokedex_entry": "It rides warm air currents.",
    "evolution_hint": "larger wings",
    "devolution_hint": "smaller crest",
}


def test_creature_plan_schema_accepts_valid_payload() -> None:
    plan = CreaturePlan.model_validate(VALID_PLAN)
    assert plan.name == "Emberwing"


def test_planner_falls_back_to_groq_after_gemini_failure() -> None:
    class FailingClient:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("503")

    class WorkingClient:
        def generate(self, prompt: str) -> str:
            import json

            return json.dumps(VALID_PLAN)

    planner = FallbackLLMPlanner(gemini_factory=lambda: FailingClient(), groq_factory=lambda: WorkingClient())
    result = planner.plan(
        CreatureInput(
            type_1="fire",
            type_2="flying",
            stats=CreatureStats(hp=70, attack=95, defense=70, special_attack=110, special_defense=75, speed=120),
            appearance_description="winged flame creature",
        )
    )
    assert result.ok
    assert result.provider_used == "groq"

