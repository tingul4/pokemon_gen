from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.generation.lora_loader import maybe_load_lora
from src.utils.config import PROJECT_ROOT, ensure_dir


@dataclass
class GenerationResult:
    image_path: str
    seed: int
    lora_used: bool
    lora_status: str


class SDXLGenerator:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        output_dir: str | Path = "outputs/generations",
        width: int = 768,
        height: int = 768,
        torch_dtype: str = "fp16",
    ) -> None:
        self.model_id = model_id
        self.output_dir = ensure_dir(output_dir)
        self.width = width
        self.height = height
        self.torch_dtype = torch_dtype
        self._pipe: Any | None = None
        self._lora_key: str | None = None
        self._lora_status = "LoRA disabled."
        self._lora_used = False

    def _load_pipeline(self, lora_path: str | None = None, lora_scale: float = 1.0) -> Any:
        import torch
        from diffusers import StableDiffusionXLPipeline

        dtype = torch.float16 if self.torch_dtype == "fp16" and torch.cuda.is_available() else torch.float32
        lora_key = f"{str(lora_path or '')}|{float(lora_scale):.6f}"
        if self._pipe is not None and lora_key != self._lora_key:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self._pipe is None:
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if dtype is torch.float16 else None,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipe.to(device)
            self._pipe.enable_attention_slicing()
            if hasattr(self._pipe.vae, "enable_slicing"):
                self._pipe.vae.enable_slicing()
            else:
                self._pipe.enable_vae_slicing()
        if lora_key != self._lora_key:
            self._lora_used, self._lora_status = maybe_load_lora(self._pipe, lora_path, lora_scale=lora_scale)
            self._lora_key = lora_key
        return self._pipe

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> GenerationResult:
        import torch

        pipe = self._load_pipeline(lora_path=lora_path, lora_scale=lora_scale)
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.width,
            height=self.height,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.png"
        path = self.output_dir / filename
        image.save(path)
        return GenerationResult(
            image_path=str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
            seed=int(seed),
            lora_used=self._lora_used,
            lora_status=self._lora_status,
        )
