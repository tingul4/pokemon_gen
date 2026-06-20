from __future__ import annotations

from pathlib import Path
from typing import Any


def maybe_load_lora(pipe: Any, lora_path: str | Path | None) -> tuple[bool, str]:
    if not lora_path:
        return False, "LoRA disabled."
    path = Path(lora_path)
    if not path.exists():
        return False, f"LoRA path does not exist: {path}"
    try:
        pipe.load_lora_weights(str(path))
        return True, f"Loaded LoRA from {path}"
    except Exception as exc:  # pragma: no cover - depends on external weights
        return False, f"Failed to load LoRA, using base SDXL: {exc}"

