from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from src.generation.ip_adapter_loader import maybe_load_ip_adapter
from src.generation.lora_loader import maybe_load_lora
from src.utils.config import PROJECT_ROOT, ensure_dir


@dataclass
class GenerationResult:
    image_path: str
    seed: int
    lora_used: bool
    lora_status: str
    ip_adapter_used: bool
    ip_adapter_status: str


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
        self._pipeline_key: str | None = None
        self._lora_status = "LoRA disabled."
        self._lora_used = False
        self._ip_adapter_status = "IP-Adapter disabled."
        self._ip_adapter_used = False

    def _load_pipeline(
        self,
        *,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
        ip_adapter_enabled: bool = False,
        ip_adapter_model_id: str | None = None,
        ip_adapter_subfolder: str | None = None,
        ip_adapter_weight_name: str | None = None,
        ip_adapter_scale: float = 0.45,
    ) -> Any:
        import torch
        from diffusers import StableDiffusionXLPipeline

        dtype = torch.float16 if self.torch_dtype == "fp16" and torch.cuda.is_available() else torch.float32
        lora_key = f"{str(lora_path or '')}|{float(lora_scale):.6f}"
        ip_key = (
            f"{str(ip_adapter_model_id or '')}|{str(ip_adapter_subfolder or '')}|"
            f"{str(ip_adapter_weight_name or '')}|{float(ip_adapter_scale):.6f}"
            if ip_adapter_enabled
            else ""
        )
        pipeline_key = f"lora={lora_key};ip={ip_key}"
        if self._pipe is not None and pipeline_key != self._pipeline_key:
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
            if not ip_adapter_enabled:
                self._pipe.enable_attention_slicing()
            if hasattr(self._pipe.vae, "enable_slicing"):
                self._pipe.vae.enable_slicing()
            else:
                self._pipe.enable_vae_slicing()
        if pipeline_key != self._pipeline_key:
            self._lora_used, self._lora_status = maybe_load_lora(self._pipe, lora_path, lora_scale=lora_scale)
            if ip_adapter_enabled:
                self._ip_adapter_used, self._ip_adapter_status = maybe_load_ip_adapter(
                    self._pipe,
                    ip_adapter_model_id,
                    subfolder=ip_adapter_subfolder,
                    weight_name=ip_adapter_weight_name,
                    scale=ip_adapter_scale,
                )
            else:
                self._ip_adapter_used = False
                self._ip_adapter_status = "IP-Adapter disabled."
            self._pipeline_key = pipeline_key
        return self._pipe

    def _open_ip_adapter_image(self, image_path: str | Path | None) -> Image.Image | None:
        if not image_path:
            self._ip_adapter_used = False
            self._ip_adapter_status = "IP-Adapter disabled: no reference image."
            return None
        path = Path(image_path)
        target = path if path.is_absolute() else PROJECT_ROOT / path
        if not target.exists():
            self._ip_adapter_used = False
            self._ip_adapter_status = f"IP-Adapter disabled: missing reference image {image_path}."
            return None
        return Image.open(target).convert("RGB")

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
        ip_adapter_enabled: bool = False,
        ip_adapter_image_path: str | Path | None = None,
        ip_adapter_model_id: str | None = None,
        ip_adapter_subfolder: str | None = None,
        ip_adapter_weight_name: str | None = None,
        ip_adapter_scale: float = 0.45,
    ) -> GenerationResult:
        import torch

        ip_adapter_image = self._open_ip_adapter_image(ip_adapter_image_path) if ip_adapter_enabled else None
        active_ip_adapter = bool(ip_adapter_enabled and ip_adapter_image is not None)
        pipe = self._load_pipeline(
            lora_path=lora_path,
            lora_scale=lora_scale,
            ip_adapter_enabled=active_ip_adapter,
            ip_adapter_model_id=ip_adapter_model_id,
            ip_adapter_subfolder=ip_adapter_subfolder,
            ip_adapter_weight_name=ip_adapter_weight_name,
            ip_adapter_scale=ip_adapter_scale,
        )
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "generator": generator,
        }
        if active_ip_adapter and self._ip_adapter_used and ip_adapter_image is not None:
            pipe_kwargs["ip_adapter_image"] = ip_adapter_image
        image = pipe(
            **pipe_kwargs,
        ).images[0]
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.png"
        path = self.output_dir / filename
        image.save(path)
        return GenerationResult(
            image_path=str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
            seed=int(seed),
            lora_used=self._lora_used,
            lora_status=self._lora_status,
            ip_adapter_used=bool(active_ip_adapter and self._ip_adapter_used),
            ip_adapter_status=self._ip_adapter_status,
        )
