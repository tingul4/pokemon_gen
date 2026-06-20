from __future__ import annotations

from typing import Any


def default_session_state() -> dict[str, Any]:
    return {
        "current_creature": None,
        "lineage_id": None,
        "last_result": None,
    }

