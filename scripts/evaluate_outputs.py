from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    demo_dir = Path("outputs/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        "| Case | Type consistency | Stat consistency | Image quality | Notes |",
        "|---|---:|---:|---:|---|",
        "| Basic fire creature | TBD | TBD | TBD | Fill after generated demo review. |",
        "| Stage 1 water/ice creature | TBD | TBD | TBD | Fill after generated demo review. |",
        "| Stage 2 dragon/electric creature | TBD | TBD | TBD | Fill after generated demo review. |",
        "| Evolution chain | TBD | TBD | TBD | Fill after generated demo review. |",
    ]
    output = demo_dir / "evaluation.md"
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"evaluation": str(output)}, indent=2))


if __name__ == "__main__":
    main()
