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
e416329 feat: implement Streamlit creature generator MVP

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
e416329 feat: implement Streamlit creature generator MVP

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
Default config uses 768 resolution, rank 8, 500 steps, and fp32 precision for stability in the local manual training loop.

### Issue / Fix
Kaggle download succeeded through the local Kaggle CLI. PokeAPI metadata was expanded to 721 records so numeric Kaggle filenames can map to structured type metadata. Initial fp16 smoke training produced a non-finite loss because the local script does not use a mixed precision scaler; the script now has a finite-loss guard and CLI override. A fp32 smoke run on GPU 0 completed 1 step with average loss `0.07429762184619904` and wrote `outputs/lora/smoke_test/pytorch_lora_weights.safetensors`. A short practical run on 256 prepared images completed 50 steps at 512px with average loss `0.02974099528975785` and wrote LoRA weights under `outputs/lora/pokecreature_sdxl_lora/`.

### Commit
22ec165 feat: validate LoRA training workflow

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
22ec165 feat: validate LoRA training workflow

## 2026-06-20 - Demo LoRA Samples

### Goal
Generate a small set of LoRA-loaded sample images for visual inspection.

### Tool / Model
Codex coding agent, SDXL base, project LoRA trained for 50 steps.

### Prompt
Generate fire, water, grass, electric, and dragon sample creatures with LoRA enabled.

### Output Summary
Generated five LoRA test images under `outputs/demo/lora_tests/` and created `outputs/demo/lora_tests/contact_sheet.png` for inspection. The short 4-step generations were nonblank; grass, electric, and dragon were more recognizable, while fire and water were more abstract due to short inference and short LoRA training.

### Human Decision
Longer training and higher inference steps are recommended for final submission material.

### Issue / Fix
`generate_sample.py` was extended with stat and output directory CLI options so demo cases can be generated without editing code.

### Commit
22ec165 feat: validate LoRA training workflow

## 2026-06-20 - Lineage Demo Validation

### Goal
Verify the required evolution and devolution demo flows outside the UI with reproducible commands.

### Tool / Model
Codex coding agent, SDXL base, 50-step project LoRA, deterministic planner fallback.

### Prompt
Generate a grass basic creature, evolve it to stage 1, generate a dragon/electric stage 2 creature, and devolve it to stage 1.

### Output Summary
Added `scripts/generate_lineage_demo.py` and generated outputs under `outputs/demo/lineage_tests/`. The summary showed `basic` BST 290, evolved `stage_1` BST 348 with parent link to the basic creature, `stage_2` BST 625, and devolved `stage_1` BST 531 with parent link to the stage 2 creature. A contact sheet was created at `outputs/demo/lineage_tests/contact_sheet.png`.

### Human Decision
Use these local outputs as demo evidence unless a longer final video is recorded separately.

### Issue / Fix
The LLM planner result is now post-processed to preserve user-requested types and deterministic/requested stage even if a provider returns inconsistent JSON.

### Commit
feat: add lineage demo validation

## 2026-06-20 - Provider Check Tooling

### Goal
Add a direct verification command for the Gemini, Groq, and Hugging Face token path requested by the human.

### Tool / Model
Codex coding agent, local shell, Python provider SDKs.

### Prompt
Continue toward the project goal and verify the `/raid/danielchen/DGM_final/.env` API key path when possible.

### Output Summary
Added `scripts/check_providers.py`. The script attempts to copy `/raid/danielchen/DGM_final/.env` into project `.env`, reports key presence without printing secrets, checks Gemini, Groq, and Hugging Face when keys exist, and runs the planner fallback path. Added a unit test proving provider JSON cannot override user types or deterministic stage.

### Human Decision
The provider check is now the canonical command for validating real API keys once the env file is visible in the runtime.

### Issue / Fix
`/raid/danielchen/DGM_final/.env` is still not visible in this environment. `python scripts/check_providers.py` reports `source_missing`, all provider keys as missing, and planner fallback as deterministic.

### Commit
feat: add provider verification script

## 2026-06-21 - Real Provider Validation

### Goal
Use the project `.env` to verify Gemini, Groq, Hugging Face, and the full planner fallback path.

### Tool / Model
Codex coding agent, `scripts/check_providers.py`, Gemini 2.5 Flash, Groq fallback model from environment, Hugging Face Hub.

### Prompt
The human reported that `.env` had been added and asked to continue testing and complete the goal.

### Output Summary
Confirmed `.env` exists with required keys set without printing secret values. Gemini direct check returned valid JSON. Groq direct check returned valid JSON. Hugging Face authentication succeeded. The full planner path first attempted Gemini, then fell back to Groq when Gemini returned a free-tier quota error. Planner output preserved the deterministic stage and user types.

### Human Decision
Treat Groq fallback behavior as verified because Gemini quota/rate failure is one of the required fallback cases.

### Issue / Fix
Real provider JSON exposed schema shape variants, such as dictionaries where strings or lists were expected. `CreaturePlan` now normalizes common provider variants before validation. Prompt constraints were also tightened to keep single-creature generation and discourage character sheets, collages, and multiple creatures.

### Commit
fix: harden real provider planning

## 2026-06-21 - Final Integration Smoke

### Goal
Verify the full runtime path after real provider validation: planner, prompt builder, LoRA loading, SDXL generation, and demo output.

### Tool / Model
Codex coding agent, Groq fallback after Gemini quota, SDXL base, 50-step LoRA.

### Prompt
Run final smoke generation with real `.env`, available GPU, LoRA, and SDXL.

### Output Summary
Generated a 20-step LoRA + SDXL image at `outputs/demo/final_smoke/20260621_090708_db44f791.png`. The image is a single fire/flying creature with visible wings, flame motifs, horns, and game-creature line art style. Full validation passed with 17 tests.

### Human Decision
Use the 20-step image as a better final smoke artifact than the earlier 1-step noise output.

### Issue / Fix
The first 20-step image resembled a character sheet with multiple creatures. Negative prompt templates now include `multiple creatures`, `character sheet`, `collage`, and `grid layout`, and the positive prompt explicitly includes `single creature`. The prompt builder was adjusted so user appearance text is preserved under the CLIP token budget.

### Commit
fix: harden real provider planning

## 2026-06-21 - Formal LoRA Comparison

### Goal
Generate a baseline image with original SDXL, run a fuller LoRA fine-tuning job, fuse the trained LoRA into the original SDXL pipeline, and compare outputs under the same condition.

### Tool / Model
Codex coding agent, CUDA GPU 0, SDXL base, diffusers LoRA fusion, Groq planner fallback.

### Prompt
The human asked to first generate one image with original SDXL using the same type, six stats, and appearance description, then formally fine-tune LoRA, merge it with the original weights, and infer again to compare visual quality.

### Output Summary
Generated a baseline original SDXL image at `outputs/demo/formal_lora_comparison/base_before_training/20260621_092059_a8ef8073.png`. Trained a new LoRA run on 256 prepared image-caption pairs for 500 steps at 512px, rank 8, fp32/no mixed precision, writing weights to `outputs/lora/pokecreature_sdxl_lora_formal_20260621_0921/`. The final average loss was `0.02772177778207697`, with checkpoints saved every 100 steps. Added `scripts/compare_base_fused_lora.py` to generate fixed-condition base SDXL versus fused LoRA comparisons and metadata. Comparison outputs were saved under `outputs/demo/formal_lora_comparison/base_vs_fused/` and `outputs/demo/formal_lora_comparison/base_vs_fused_scale05/`.

### Human Decision
Use `lora_scale=0.5` as the stronger demo comparison because it preserved the condition while producing a more stable fused output than `lora_scale=0.8`.

### Issue / Fix
The training command initially emitted an HF token warning because `scripts/train_lora_sdxl.py` did not load `.env`. The script now calls `load_environment()` before loading SDXL. Visual inspection showed `lora_scale=0.8` made the result more dataset-like but introduced stronger anatomy artifacts, while `lora_scale=0.5` provided a better balance between prompt adherence and LoRA style.

### Commit
feat: add fused LoRA comparison workflow
