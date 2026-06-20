from __future__ import annotations

from pathlib import Path

from PIL import Image


def save_image(image: Image.Image, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target

