from __future__ import annotations

from src.llm.client import FallbackLLMPlanner
from src.llm.schemas import CreatureInput, PlannedCreatureResult


def plan_creature(creature_input: CreatureInput) -> PlannedCreatureResult:
    return FallbackLLMPlanner().plan(creature_input)

