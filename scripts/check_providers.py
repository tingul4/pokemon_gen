from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import FallbackLLMPlanner
from src.llm.gemini_client import GeminiPlannerClient
from src.llm.groq_client import GroqPlannerClient
from src.llm.schemas import CreatureInput, CreatureStats
from src.utils.config import PROJECT_ROOT, load_environment


DEFAULT_ENV_SOURCE = Path("/raid/danielchen/DGM_final/.env")
ENV_KEYS = ("GEMINI_API_KEY", "GROQ_API_KEY", "GROQ_FALLBACK_MODEL", "HF_TOKEN")


def copy_env_if_available(source: Path, target: Path) -> dict[str, Any]:
    if not source.exists():
        return {"copied": False, "source": str(source), "reason": "source_missing"}
    shutil.copy2(source, target)
    target.chmod(0o600)
    return {"copied": True, "source": str(source), "target": str(target)}


def env_status() -> dict[str, str]:
    return {key: "set" if os.getenv(key) else "missing" for key in ENV_KEYS}


def _parse_json_response(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def check_gemini() -> dict[str, Any]:
    if not os.getenv("GEMINI_API_KEY"):
        return {"ok": False, "skipped": True, "reason": "GEMINI_API_KEY missing"}
    try:
        text = GeminiPlannerClient().generate('Return exactly this JSON object: {"ok": true, "provider": "gemini"}')
        return {"ok": bool(text), "json": _parse_json_response(text), "chars": len(text)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_groq() -> dict[str, Any]:
    if not os.getenv("GROQ_API_KEY"):
        return {"ok": False, "skipped": True, "reason": "GROQ_API_KEY missing"}
    try:
        text = GroqPlannerClient().generate('Return exactly this JSON object: {"ok": true, "provider": "groq"}')
        return {"ok": bool(text), "json": _parse_json_response(text), "chars": len(text)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_hf() -> dict[str, Any]:
    token = os.getenv("HF_TOKEN")
    if not token:
        return {"ok": False, "skipped": True, "reason": "HF_TOKEN missing"}
    try:
        from huggingface_hub import HfApi

        whoami = HfApi(token=token).whoami()
        return {"ok": True, "name": whoami.get("name") or whoami.get("fullname") or "authenticated"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_planner() -> dict[str, Any]:
    creature_input = CreatureInput(
        type_1="fire",
        type_2="flying",
        stats=CreatureStats(hp=70, attack=95, defense=70, special_attack=110, special_defense=75, speed=120),
        appearance_description="a small dragon-like creature with feathered wings and a glowing flame crest",
    )
    result = FallbackLLMPlanner().plan(creature_input)
    return {
        "ok": result.ok,
        "provider_used": result.provider_used,
        "warnings": result.warnings,
        "stage": result.plan.evolution_stage if result.plan else None,
        "types": result.plan.types if result.plan else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-env-from", default=str(DEFAULT_ENV_SOURCE))
    parser.add_argument("--target-env", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--no-network", action="store_true", help="Only report env key presence without API calls.")
    args = parser.parse_args()

    copy_result: dict[str, Any] | None = None
    if not args.no_copy:
        copy_result = copy_env_if_available(Path(args.copy_env_from), Path(args.target_env))
    load_environment(args.target_env)

    report: dict[str, Any] = {"env_copy": copy_result, "env": env_status()}
    if not args.no_network:
        report["providers"] = {
            "gemini": check_gemini(),
            "groq": check_groq(),
            "huggingface": check_hf(),
            "planner": check_planner(),
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

