from __future__ import annotations

from typing import Any

import streamlit as st

from src.llm.schemas import POKEMON_TYPES


def type_selectors() -> tuple[str, str | None]:
    type_1 = st.selectbox("Type 1", POKEMON_TYPES, index=POKEMON_TYPES.index("fire"))
    type_2_options = ["none"] + POKEMON_TYPES
    type_2 = st.selectbox("Type 2", type_2_options, index=0)
    return type_1, None if type_2 == "none" else type_2


def stat_sliders(defaults: dict[str, int] | None = None) -> dict[str, int]:
    defaults = defaults or {}
    cols = st.columns(3)
    names = [
        ("hp", "HP"),
        ("attack", "Attack"),
        ("defense", "Defense"),
        ("special_attack", "Special Attack"),
        ("special_defense", "Special Defense"),
        ("speed", "Speed"),
    ]
    values: dict[str, int] = {}
    for idx, (key, label) in enumerate(names):
        with cols[idx % 3]:
            values[key] = st.slider(label, 1, 180, int(defaults.get(key, 70)))
    return values


def render_creature(creature: dict[str, Any]) -> None:
    image_path = creature.get("image_path")
    if image_path:
        st.image(image_path, caption=creature.get("name", "Generated creature"))
    st.subheader(creature.get("name", "Unnamed creature"))
    st.write(f"Types: {', '.join(creature.get('types', []))}")
    st.write(f"Evolution stage: `{creature.get('stage')}`")
    st.write(f"Base stat total: `{sum(creature.get('stats', {}).values())}`")
    st.json(creature.get("stats", {}))
    st.write(creature.get("pokedex_entry", ""))

