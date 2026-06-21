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


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(_stringify(item) for item in value if item is not None)
    if isinstance(value, dict):
        preferred = value.get("description") or value.get("text") or value.get("prompt")
        parts: list[str] = []
        if preferred:
            parts.append(_stringify(preferred))
        for key, item in value.items():
            if key in {"description", "text", "prompt"}:
                continue
            text = _stringify(item)
            if text:
                parts.append(text)
        return ", ".join(parts)
    return str(value)


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_stringify(item).strip() for item in value if _stringify(item).strip()]
    if isinstance(value, dict):
        items: list[str] = []
        for item in value.values():
            items.extend(_listify(item))
        return items
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


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

    @field_validator(
        "name",
        "visual_concept",
        "sdxl_prompt",
        "negative_prompt",
        "pokedex_entry",
        "evolution_hint",
        "devolution_hint",
        mode="before",
    )
    @classmethod
    def coerce_string_fields(cls, value: Any) -> str:
        return _stringify(value)

    @field_validator("core_motifs", "color_palette", mode="before")
    @classmethod
    def coerce_list_fields(cls, value: Any) -> list[str]:
        return _listify(value)

    @field_validator("stat_interpretation", mode="before")
    @classmethod
    def coerce_stat_interpretation(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {name: _stringify(value) for name in REQUIRED_STATS}
        return {str(key): _stringify(item) for key, item in value.items()}


class PlannedCreatureResult(BaseModel):
    ok: bool
    plan: CreaturePlan | None = None
    provider_used: str | None = None
    raw_response: Any = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
