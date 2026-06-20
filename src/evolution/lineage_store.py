from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.utils.config import PROJECT_ROOT, ensure_dir


@dataclass
class LineageStore:
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "lineages")

    def __post_init__(self) -> None:
        self.output_dir = ensure_dir(self.output_dir)

    def new_lineage_id(self) -> str:
        return f"ln_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

    def new_creature_id(self) -> str:
        return f"cr_{uuid4().hex[:10]}"

    def path_for(self, lineage_id: str) -> Path:
        return self.output_dir / f"{lineage_id}.json"

    def load(self, lineage_id: str) -> dict[str, Any]:
        with self.path_for(lineage_id).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_creature(self, lineage_id: str | None, creature: dict[str, Any]) -> dict[str, Any]:
        lineage_id = lineage_id or self.new_lineage_id()
        path = self.path_for(lineage_id)
        if path.exists():
            payload = self.load(lineage_id)
        else:
            payload = {"lineage_id": lineage_id, "creatures": []}
        creature = dict(creature)
        creature.setdefault("creature_id", self.new_creature_id())
        creature.setdefault("lineage_id", lineage_id)
        payload["creatures"].append(creature)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return creature

