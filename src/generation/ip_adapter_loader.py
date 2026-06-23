from __future__ import annotations

from typing import Any


def maybe_load_ip_adapter(
    pipe: Any,
    model_id: str | None,
    *,
    subfolder: str | None,
    weight_name: str | None,
    scale: float = 0.45,
) -> tuple[bool, str]:
    if not model_id or not weight_name:
        return False, "IP-Adapter disabled."
    try:
        pipe.load_ip_adapter(
            model_id,
            subfolder=subfolder or "",
            weight_name=weight_name,
        )
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(float(scale))
        return True, f"Loaded IP-Adapter {model_id}/{weight_name} at scale {float(scale):.2f}"
    except Exception as exc:  # pragma: no cover - depends on local model/cache/network state.
        return False, f"IP-Adapter load failed: {exc}"
