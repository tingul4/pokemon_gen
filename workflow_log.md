# Workflow Log

## 2026-06-20 - Project Scaffold and MVP Implementation

### Goal
Complete the project according to `plan.md`: Streamlit app, LLM planner, SDXL generation, LoRA training pipeline, dataset preparation, evolution/devolution, tests, README, and workflow log.

### Tool / Model
Codex coding agent.

### Prompt
The human asked to complete the project from `plan.md`, copy API keys from `/raid/danielchen/DGM_final/.env` for testing, select an available GPU, use the Kaggle Pokemon image dataset, build fine-tuning code, start training when ready, and observe metrics.

### Output Summary
Created the repository structure, configuration files, environment examples, deterministic stage estimator, LLM schemas and Gemini-to-Groq fallback planner, SDXL prompt builder, SDXL generation wrapper, LoRA loader, PokeAPI fetcher, Kaggle dataset preparation pipeline, caption builder, lineage storage, evolution/devolution planner, Streamlit MVP, sample generation script, LoRA training script, evaluation template, and unit tests.

### Human Decision
The requested technical route was preserved: Streamlit, Gemini 2.5 Flash, Groq fallback, SDXL, diffusers, LoRA, Kaggle images, and PokeAPI metadata.

### Issue / Fix
The repository was initially not a git repository and only contained `AGENTS.md`, `plan.md`, and `requirements.txt`. The implementation created the missing scaffold and code paths, initialized git, and verified import, compile, and unit test paths.

### Commit
Pending.

## 2026-06-20 - LLM Fallback Design

### Goal
Implement structured JSON planning with provider fallback and graceful failure behavior.

### Tool / Model
Codex coding agent.

### Prompt
Implement Gemini 2.5 Flash primary planner with Groq fallback, schema validation, JSON repair, and deterministic fallback.

### Output Summary
Added Pydantic schemas, Gemini and Groq clients, prompt construction, JSON extraction, repair prompt, provider fallback ordering, and deterministic fallback plan generation.

### Human Decision
Fallback behavior is kept visible in the Streamlit debug panel through `llm_provider` and warnings.

### Issue / Fix
API failures or missing keys should not crash the app. The planner returns a deterministic structured plan after both providers fail.

### Commit
Pending.

## 2026-06-20 - LoRA Training Setup

### Goal
Provide code that can train SDXL LoRA from prepared image-caption pairs.

### Tool / Model
Codex coding agent.

### Prompt
Build fine-tuning code and start training when ready, observing training metrics.

### Output Summary
Added `configs/lora_sdxl.yaml` and `scripts/train_lora_sdxl.py`. The script loads SDXL, freezes base components, adds LoRA adapters to UNet attention layers, trains on `data/processed/captions.jsonl`, saves checkpoints, and writes `training_metrics.json`.

### Human Decision
Default config uses 768 resolution, rank 8, and 500 steps to reduce GPU memory risk.

### Issue / Fix
Kaggle download succeeded through the local Kaggle CLI. PokeAPI metadata was expanded to 721 records so numeric Kaggle filenames can map to structured type metadata. Initial fp16 smoke training produced a non-finite loss because the local script does not use a mixed precision scaler; the script now has a finite-loss guard and CLI override. A fp32 smoke run on GPU 0 completed 1 step with average loss `0.07429762184619904` and wrote `outputs/lora/smoke_test/pytorch_lora_weights.safetensors`.

### Commit
Pending.

## 2026-06-20 - Validation and Inference Smoke Tests

### Goal
Verify the current MVP code paths with local commands.

### Tool / Model
Codex coding agent with local shell, uv, CUDA GPU 0, Streamlit, PyTorch, and diffusers.

### Prompt
Run the relevant checks, choose an available GPU, prepare dataset samples, train briefly, and observe metrics.

### Output Summary
Created `.venv` with Python 3.11.13 using uv. Installed requirements. GPU check showed GPU 0 was the better choice: about 26GB / 98GB memory used and 9% utilization when checked. `pytest` passed 15 tests. Streamlit launched successfully on port 8501 in headless smoke mode. Base SDXL generated an image with `python scripts/generate_sample.py --steps 1`. LoRA loading generated an image with the smoke LoRA weights. Prompt builder was tightened after SDXL warned that long prompts were truncated by CLIP.

### Human Decision
Use GPU 0 for smoke tests. Full quality LoRA training remains a longer run using `configs/lora_sdxl.yaml`.

### Issue / Fix
The source `.env` path `/raid/danielchen/DGM_final/.env` was not present in this environment, so API keys could not be copied. The planner was verified through deterministic fallback. Kaggle credentials were still available to the Kaggle CLI, so the dataset download succeeded.

### Commit
Pending.
