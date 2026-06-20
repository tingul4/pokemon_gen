from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment(env_path: str | Path | None = None) -> None:
    """Load project .env without failing when it is absent."""
    candidate = Path(env_path) if env_path else PROJECT_ROOT / ".env"
    if candidate.exists():
        load_dotenv(candidate)
    else:
        load_dotenv()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.mkdir(parents=True, exist_ok=True)
    return target

