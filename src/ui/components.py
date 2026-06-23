from __future__ import annotations

from typing import Any

import streamlit as st

from src.llm.schemas import POKEMON_TYPES
from src.ui.state import stage_label


TYPE_LABELS = {
    "normal": "一般",
    "fire": "火",
    "water": "水",
    "grass": "草",
    "electric": "電",
    "ice": "冰",
    "fighting": "格鬥",
    "poison": "毒",
    "ground": "地面",
    "flying": "飛行",
    "psychic": "超能力",
    "bug": "蟲",
    "rock": "岩石",
    "ghost": "幽靈",
    "dragon": "龍",
    "dark": "惡",
    "steel": "鋼",
    "fairy": "妖精",
}


def type_label(type_name: str) -> str:
    return TYPE_LABELS.get(type_name, type_name)


def type_selectors() -> tuple[str, str | None]:
    type_1 = st.selectbox("屬性 1", POKEMON_TYPES, index=POKEMON_TYPES.index("fire"), format_func=type_label)
    type_2_options = ["none"] + POKEMON_TYPES
    type_2 = st.selectbox(
        "屬性 2（選填）",
        type_2_options,
        index=0,
        format_func=lambda value: "無" if value == "none" else type_label(value),
    )
    return type_1, None if type_2 == "none" else type_2


def stat_sliders(defaults: dict[str, int] | None = None) -> dict[str, int]:
    defaults = defaults or {}
    cols = st.columns(3)
    names = [
        ("hp", "HP"),
        ("attack", "攻擊"),
        ("defense", "防禦"),
        ("special_attack", "特攻"),
        ("special_defense", "特防"),
        ("speed", "速度"),
    ]
    values: dict[str, int] = {}
    for idx, (key, label) in enumerate(names):
        with cols[idx % 3]:
            values[key] = st.slider(label, 1, 180, int(defaults.get(key, 70)))
    return values


def render_creature(creature: dict[str, Any]) -> None:
    image_path = creature.get("image_path")
    if image_path:
        st.image(image_path, caption=creature.get("name", "生成結果"))
    st.subheader(creature.get("name", "未命名怪獸"))
    type_text = " / ".join(type_label(item) for item in creature.get("types", []))
    st.write(f"屬性：{type_text}")
    st.write(f"進化階段：{stage_label(creature.get('stage'))}")
    st.write(f"六圍總和：`{sum(creature.get('stats', {}).values())}`")
    stats = creature.get("stats", {})
    st.table(
        {
            "項目": ["HP", "攻擊", "防禦", "特攻", "特防", "速度"],
            "數值": [
                stats.get("hp"),
                stats.get("attack"),
                stats.get("defense"),
                stats.get("special_attack"),
                stats.get("special_defense"),
                stats.get("speed"),
            ],
        }
    )
    st.write(creature.get("pokedex_entry", ""))
