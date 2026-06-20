# Conditional Pokemon-style Creature Generator

## Overview

This repository is an academic Deep Generative Models course project. It builds a Streamlit app that generates original Pokemon-style creature concepts from structured inputs: one or two elemental types, six base stats, and an appearance description.

The project generates original fantasy creatures only. It is not affiliated with Nintendo, Game Freak, Creatures Inc., or The Pokemon Company.

## Features

- Streamlit input form for types, stats, and appearance text
- Gemini 2.5 Flash planner with Groq fallback
- Pydantic validation for structured LLM JSON
- Deterministic fallback planner when API calls fail
- SDXL image generation with optional LoRA loading
- Dataset preparation for Kaggle Pokemon image datasets
- PokeAPI metadata fetcher
- SDXL LoRA training script
- Evolution and devolution planning using saved lineage metadata
- Debug panel with provider, prompt, seed, LoRA status, and raw JSON

## System Architecture

User input flows through deterministic stat validation, evolution stage estimation, LLM planning, prompt building, SDXL image generation, lineage storage, and Streamlit result display.

## Tech Stack

- Python 3.11
- Streamlit
- Pydantic
- Gemini 2.5 Flash
- Groq fallback model configured by `GROQ_FALLBACK_MODEL`
- PyTorch
- Hugging Face Diffusers
- SDXL
- PEFT LoRA
- PokeAPI and Kaggle dataset tooling

## Dataset

Raw datasets are not committed. Download the Kaggle dataset manually or through Kaggle credentials:

[Pokemon Images Dataset](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset/data)

Place images under:

```bash
data/raw/kaggle_pokemon_images/
```

PokeAPI metadata can be fetched with:

```bash
python scripts/fetch_pokeapi.py --limit 150
```

## Installation

Use uv with Python 3.11:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill:

```bash
GEMINI_API_KEY=
GROQ_API_KEY=
GROQ_FALLBACK_MODEL=
HF_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
```

The app handles missing keys by showing warnings and using the deterministic fallback planner.

Provider smoke check:

```bash
python scripts/check_providers.py
```

By default this tries to copy `/raid/danielchen/DGM_final/.env` into the project `.env`, reports which keys are present, checks Gemini, Groq, and Hugging Face authentication, then runs the full planner fallback path. It never prints secret values.

## Data Preparation

Fetch metadata:

```bash
python scripts/fetch_pokeapi.py --limit 150
```

Prepare LoRA image-caption pairs:

```bash
python scripts/prepare_lora_dataset.py --max-images 500 --resolution 768
```

Outputs:

- `data/processed/metadata.json`
- `data/processed/captions.jsonl`
- `data/processed/lora_images/`
- `data/samples/captions_sample.jsonl`

## SDXL + LoRA

Generate a base SDXL sample:

```bash
python scripts/generate_sample.py --steps 10
```

Train LoRA after dataset preparation:

```bash
accelerate launch scripts/train_lora_sdxl.py --config configs/lora_sdxl.yaml
```

The default config uses fp32 precision (`mixed_precision: "no"`) because the local manual training loop produced stable finite loss in fp32. If GPU memory is constrained, reduce `resolution`, `rank`, or `max_train_steps` before trying fp16.

Quick smoke test on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_sdxl.py \
  --config configs/lora_sdxl.yaml \
  --max-train-steps 1 \
  --resolution 512 \
  --output-dir outputs/lora/smoke_test \
  --train-batch-size 1 \
  --mixed-precision no
```

The default config writes weights and metrics to:

```bash
outputs/lora/pokecreature_sdxl_lora/
```

Run inference with LoRA:

```bash
python scripts/generate_sample.py --use-lora --lora-path outputs/lora/pokecreature_sdxl_lora
```

Generate a small evolution/devolution demo set:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_lineage_demo.py \
  --steps 4 \
  --use-lora \
  --lora-path outputs/lora/pokecreature_sdxl_lora \
  --output-dir outputs/demo/lineage_tests
```

## Running the App

```bash
streamlit run app.py
```

Example input:

- Type 1: `fire`
- Type 2: `flying`
- HP 70, Attack 95, Defense 70, Special Attack 110, Special Defense 75, Speed 120
- Appearance: `a small dragon-like creature with feathered wings and a glowing flame crest`

Expected behavior: the app estimates `stage_2`, plans metadata, generates an image, saves lineage metadata, and enables devolution.

## Evolution and Devolution

Stage estimation is deterministic:

- `basic`: BST <= 340
- `stage_1`: 341 <= BST <= 480
- `stage_2`: BST > 480

Evolution and devolution use stored lineage metadata to preserve motifs, type identity, color palette, and parent-child links.

## Evaluation

Create an evaluation template:

```bash
python scripts/evaluate_outputs.py
```

Manual demo assets should be saved under:

```bash
outputs/demo/
```

Current local demo outputs include:

- `outputs/demo/lora_tests/contact_sheet.png`
- `outputs/demo/lineage_tests/contact_sheet.png`
- `outputs/demo/lineage_tests/lineage_demo_summary.json`

Validation performed during setup:

- `python -m compileall src scripts app.py`
- `pytest -q`
- `python scripts/fetch_pokeapi.py --limit 721`
- `python scripts/prepare_lora_dataset.py --max-images 256 --resolution 512`
- `CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_sdxl.py --max-train-steps 50 --resolution 512 --mixed-precision no`
- `CUDA_VISIBLE_DEVICES=0 python scripts/generate_sample.py --steps 1`
- `CUDA_VISIBLE_DEVICES=0 python scripts/generate_sample.py --steps 4 --use-lora --lora-path outputs/lora/pokecreature_sdxl_lora`
- `CUDA_VISIBLE_DEVICES=0 python scripts/generate_lineage_demo.py --steps 4 --use-lora --lora-path outputs/lora/pokecreature_sdxl_lora`
- `python scripts/check_providers.py`

## Agent Workflow Summary

Development followed the milestone plan:

1. Project scaffold and environment setup
2. PokeAPI and Kaggle dataset preparation
3. Deterministic evolution stage estimator
4. Gemini to Groq fallback planner
5. SDXL prompt builder and generation wrapper
6. LoRA training and inference integration
7. Streamlit MVP and lineage controls
8. Tests, README, and workflow log

## Limitations

- SDXL and LoRA training require a CUDA GPU for practical runtime.
- LoRA quality depends on dataset cleanliness and training duration.
- LLM provider availability depends on API quota and model access.
- Generated images should be manually reviewed to avoid close resemblance to official characters.

## Future Work

- Add gallery browsing and saved lineage reload UI
- Add radar chart visualization for stats
- Add demo screenshot automation
- Add stronger image quality evaluation
