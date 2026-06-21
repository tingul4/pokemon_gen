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
- Dataset preparation for the Kaggle Complete Pokemon Image Dataset
- Complete Pokedex CSV dataset preparation
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
- cristobalmitchell/pokedex dataset tooling

## Dataset

Raw datasets are not committed. Use the Complete Pokedex Dataset, available from GitHub and Kaggle:

- [GitHub: cristobalmitchell/pokedex](https://github.com/cristobalmitchell/pokedex)
- [Kaggle: The Complete Pokedex Dataset](https://www.kaggle.com/cristobalmitchell/pokedex)

Place or clone the dataset under:

```bash
data/raw/cristobalmitchell_pokedex/
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

Download or clone the dataset:

```bash
git clone https://github.com/cristobalmitchell/pokedex.git data/raw/cristobalmitchell_pokedex
```

Prepare LoRA image-caption pairs and structured annotations:

```bash
python scripts/prepare_lora_dataset.py --resolution 512
```

Outputs:

- `data/processed/metadata.json`
- `data/processed/captions.jsonl`
- `data/processed/annotations.jsonl`
- `data/processed/lora_images/`
- `data/samples/captions_sample.jsonl`
- `data/samples/annotations_sample.jsonl`

`captions.jsonl` is kept compact for SDXL CLIP token limits. `annotations.jsonl` preserves the richer label for each processed image, including source Pokemon name, form name, national number, types, HP, Attack, Defense, Special Attack, Special Defense, Speed, base stat total, abilities, height, weight, official description, classification, caption, and appearance description.

The appearance description is built directly from the dataset CSV:

- classification from the official Pokedex source
- official website description
- type, six base stats, abilities, height, and weight
- evolution-chain fields and special-form flags in annotations

LoRA training captions intentionally stay shorter and more generic. They use the style token, type, compact official species profile terms, and numeric stats, but do not include long official Pokédex text, official character names, or inferred stat-to-visual traits.

At inference time the Streamlit app still builds a richer SDXL prompt from the user's inputs and the LLM planner. When LoRA is enabled, the prompt includes `pokecreature_style` and the same compact stats token format used in training, such as `stats hp70 atk95 def70 spa110 spd75 spe120`, so the LoRA sees consistent style/type/stat conditioning.

## SDXL + LoRA

Generate a base SDXL sample:

```bash
python scripts/generate_sample.py --steps 10
```

Train LoRA after dataset preparation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_sdxl.py --config configs/lora_sdxl.yaml
```

The default config uses 512px fp32 training for 1500 steps (`mixed_precision: "no"`) because the local manual training loop produced stable finite loss in fp32. If GPU memory is constrained, reduce `resolution`, `rank`, or `max_train_steps` before trying fp16.

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
outputs/lora/pokecreature_sdxl_lora_pokedex/
```

Run inference with LoRA:

```bash
python scripts/generate_sample.py --use-lora --lora-path outputs/lora/pokecreature_sdxl_lora_pokedex
```

Compare original SDXL with a fused LoRA result under the same condition:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/compare_base_fused_lora.py \
  --type-1 fire \
  --type-2 flying \
  --appearance "a small dragon-like creature with feathered wings and a glowing flame crest" \
  --hp 70 \
  --attack 95 \
  --defense 70 \
  --special-attack 110 \
  --special-defense 75 \
  --speed 120 \
  --seed 12345 \
  --steps 20 \
  --lora-path outputs/lora/pokecreature_sdxl_lora_pokedex \
  --lora-scale 0.3 \
  --output-dir outputs/demo/pokedex_dataset_comparison/base_vs_fused_scale03
```

Generate a small evolution/devolution demo set:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_lineage_demo.py \
  --steps 4 \
  --use-lora \
  --lora-path outputs/lora/pokecreature_sdxl_lora_pokedex \
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

- `outputs/demo/pokedex_dataset_comparison/base_vs_fused_scale03/*_comparison.png`

Validation performed during setup:

- `python -m compileall src scripts app.py`
- `pytest -q`
- `python scripts/check_providers.py --no-copy`
- `python scripts/prepare_lora_dataset.py --resolution 512`
- `CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_sdxl.py --max-train-steps 1500 --resolution 512 --mixed-precision no`
- `CUDA_VISIBLE_DEVICES=0 python scripts/generate_sample.py --steps 1`
- `CUDA_VISIBLE_DEVICES=0 python scripts/generate_sample.py --steps 4 --use-lora --lora-path outputs/lora/pokecreature_sdxl_lora_pokedex`
- `CUDA_VISIBLE_DEVICES=0 python scripts/generate_lineage_demo.py --steps 4 --use-lora --lora-path outputs/lora/pokecreature_sdxl_lora_pokedex`
- `python scripts/check_providers.py`
- `CUDA_VISIBLE_DEVICES=0 python scripts/compare_base_fused_lora.py --lora-path outputs/lora/pokecreature_sdxl_lora_pokedex --lora-scale 0.3`

Provider validation with the local `.env` showed all required keys are present. Groq and Hugging Face checks succeeded. Gemini was reachable earlier in validation, then hit the free-tier quota limit during later checks; the planner successfully used Groq fallback for that required failure case.

Current dataset validation uses cristobalmitchell/pokedex CSV descriptions and stats directly, without PokeAPI metadata matching. The rebuilt processed dataset contains 1137 image-caption pairs from `large_images` and `alt_images`, 898 metadata records, 898 unique names, no unmatched image files, no JSON Unicode escapes, and no captions over the 77-token CLIP limit. The full-data 1500-step LoRA run completed at 512px/rank 8 with average loss `0.0632460796807427`, writing weights to `outputs/lora/pokecreature_sdxl_lora_pokedex_20260621_2346/`. The fixed-condition fused comparison is saved at `outputs/demo/pokedex_dataset_comparison/base_vs_fused_scale03/20260621_235736_cdd80921_comparison.png`.

## Agent Workflow Summary

Development followed the milestone plan:

1. Project scaffold and environment setup
2. Complete Pokedex dataset preparation
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
