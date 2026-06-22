from __future__ import annotations

from pathlib import Path

from src.generation.lora_loader import maybe_load_lora


class FakePipe:
    def __init__(self) -> None:
        self.loaded_path: str | None = None
        self.fused_scale: float | None = None

    def load_lora_weights(self, path: str) -> None:
        self.loaded_path = path

    def fuse_lora(self, lora_scale: float) -> None:
        self.fused_scale = lora_scale


def test_maybe_load_lora_fuses_with_requested_scale(tmp_path: Path) -> None:
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    pipe = FakePipe()

    used, status = maybe_load_lora(pipe, lora_dir, lora_scale=0.5)

    assert used is True
    assert pipe.loaded_path == str(lora_dir)
    assert pipe.fused_scale == 0.5
    assert "scale 0.50" in status


def test_maybe_load_lora_reports_missing_path(tmp_path: Path) -> None:
    used, status = maybe_load_lora(FakePipe(), tmp_path / "missing", lora_scale=0.5)

    assert used is False
    assert "does not exist" in status
