from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch_pokeapi import fetch_pokemon_metadata, save_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    records = fetch_pokemon_metadata(limit=args.limit)
    output = save_metadata(records)
    print(f"Saved {len(records)} records to {output}")


if __name__ == "__main__":
    main()
