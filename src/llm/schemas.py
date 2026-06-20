from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.evolution.stage_estimator import REQUIRED_STATS


POKEMON_TYPES = [
    "normal",
    "fire",
    "water",
    "grass",
    "electric",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
]


class CreatureStats(BaseModel):
    hp: int = Field(ge=1, le=255)
    attack: int = Field(ge=1, le=255)
    defense: int = Field(ge=1, le=255)
    special_attack: int = Field(ge=1, le=255)
    special_defense: int = Field(ge=1, le=255)
    speed: int = Field(ge=1, le=255)

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in REQUIRED_STATS}


class CreatureInput(BaseModel):
    type_1: str
    type_2: str | None = None
    stats: CreatureStats
    appearance_description: str
    requested_stage: Literal["basic", "stage_1", "stage_2"] | None = None
    lineage_constraints: str | None = None

    @field_validator("type_1", "type_2")
    @classmethod
    def validate_type(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        normalized = value.lower().strip()
        if normalized not in POKEMON_TYPES:
            raise ValueError(f"Unsupported type: {value}")
        return normalized

    @property
    def types(self) -> list[str]:
        values = [self.type_1]
        if self.type_2 and self.type_2 != self.type_1:
            values.append(self.type_2)
        return values


class CreaturePlan(BaseModel):
    name: str
    types: list[str]
    evolution_stage: Literal["basic", "stage_1", "stage_2"]
    visual_concept: str
    stat_interpretation: dict[str, str]
    core_motifs: list[str]
    color_palette: list[str]
    sdxl_prompt: str
    negative_prompt: str
    pokedex_entry: str
    evolution_hint: str
    devolution_hint: str

    @field_validator("types")
    @classmethod
    def validate_types(cls, value: list[str]) -> list[str]:
        if not value or len(value) > 2:
            raise ValueError("CreaturePlan.types must contain one or two types.")
        return [item.lower().strip() for item in value]


class PlannedCreatureResult(BaseModel):
    ok: bool
    plan: CreaturePlan | None = None
    provider_used: str | None = None
    raw_response: Any = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)

