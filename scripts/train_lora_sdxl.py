from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.config import load_environment


class CaptionImageDataset(Dataset):
    def __init__(self, captions_file: Path, resolution: int) -> None:
        self.rows = [json.loads(line) for line in captions_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))
        tensor = tensor.permute(2, 0, 1) / 127.5 - 1.0
        return {"pixel_values": tensor, "text": row["text"]}


def _load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _save_lora_weights(pipe, unet, output_dir: Path) -> None:
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import get_peft_model_state_dict

    output_dir.mkdir(parents=True, exist_ok=True)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_lora_state_dict,
    )


def train(config: dict) -> None:
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from peft import LoraConfig

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if config.get("mixed_precision") == "fp16" and device.type == "cuda" else torch.float32
    captions_file = Path(config["captions_file"])
    if not captions_file.exists():
        raise FileNotFoundError(f"Missing captions file: {captions_file}. Run scripts/prepare_lora_dataset.py first.")

    dataset = CaptionImageDataset(captions_file, int(config.get("resolution", 768)))
    if len(dataset) == 0:
        raise RuntimeError("No training rows found.")
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=0,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        config["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype is torch.float16 else None,
    )
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet.train()

    lora_config = LoraConfig(
        r=int(config.get("rank", 8)),
        lora_alpha=int(config.get("rank", 8)),
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(lora_config)
    trainable_params = [param for param in pipe.unet.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(config.get("learning_rate", 1e-4)))
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    max_steps = int(config.get("max_train_steps", 500))
    grad_accum = int(config.get("gradient_accumulation_steps", 4))
    checkpoint_every = int(config.get("checkpointing_steps", 100))
    output_dir = Path(config.get("output_dir", "outputs/lora/pokecreature_sdxl_lora_pokedex"))
    global_step = 0
    running_loss = 0.0
    progress = tqdm(total=max_steps, desc="Training SDXL LoRA")

    while global_step < max_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            prompts = list(batch["text"])
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                    prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                add_time_ids = pipe._get_add_time_ids(
                    (config["resolution"], config["resolution"]),
                    (0, 0),
                    (config["resolution"], config["resolution"]),
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
                ).to(device)
                add_time_ids = add_time_ids.repeat(latents.shape[0], 1)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = pipe.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float()) / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected. Try --mixed-precision no for a smoke test, "
                    "or reduce learning rate/resolution for fp16 training."
                )
            loss.backward()
            running_loss += float(loss.item()) * grad_accum

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=running_loss / max(global_step, 1))

            if checkpoint_every and global_step % checkpoint_every == 0:
                _save_lora_weights(pipe, pipe.unet, output_dir / f"checkpoint-{global_step}")
            if global_step >= max_steps:
                break

    progress.close()
    _save_lora_weights(pipe, pipe.unet, output_dir)
    metrics = {
        "max_train_steps": max_steps,
        "average_loss": running_loss / max(global_step, 1),
        "resolution": config.get("resolution"),
        "rank": config.get("rank"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_sdxl.yaml")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=["no", "fp16"], default=None)
    args = parser.parse_args()
    load_environment()
    config = _load_config(args.config)
    if args.max_train_steps is not None:
        config["max_train_steps"] = args.max_train_steps
    if args.resolution is not None:
        config["resolution"] = args.resolution
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.train_batch_size is not None:
        config["train_batch_size"] = args.train_batch_size
    if args.mixed_precision is not None:
        config["mixed_precision"] = args.mixed_precision
    train(config)


if __name__ == "__main__":
    main()
