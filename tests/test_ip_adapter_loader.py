from __future__ import annotations

from src.generation.ip_adapter_loader import maybe_load_ip_adapter


class FakePipe:
    def __init__(self) -> None:
        self.loaded: tuple[str, str, str] | None = None
        self.scale: float | None = None

    def load_ip_adapter(self, model_id: str, *, subfolder: str, weight_name: str) -> None:
        self.loaded = (model_id, subfolder, weight_name)

    def set_ip_adapter_scale(self, scale: float) -> None:
        self.scale = scale


def test_maybe_load_ip_adapter_sets_scale() -> None:
    pipe = FakePipe()

    used, status = maybe_load_ip_adapter(
        pipe,
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
        scale=0.45,
    )

    assert used is True
    assert pipe.loaded == ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.bin")
    assert pipe.scale == 0.45
    assert "scale 0.45" in status


def test_maybe_load_ip_adapter_reports_disabled_without_weight() -> None:
    used, status = maybe_load_ip_adapter(FakePipe(), "h94/IP-Adapter", subfolder="sdxl_models", weight_name=None)

    assert used is False
    assert "disabled" in status
