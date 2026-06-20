from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare_dataset import prepare_lora_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-image-dir", default="data/raw/kaggle_pokemon_images")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=768)
    args = parser.parse_args()
    rows = prepare_lora_dataset(
        raw_image_dir=args.raw_image_dir,
        max_images=args.max_images,
        resolution=args.resolution,
    )
    print(f"Prepared {len(rows)} image-caption pairs.")


if __name__ == "__main__":
    main()
