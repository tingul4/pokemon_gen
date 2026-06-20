# plan.md

## Project Goal

Build a working Streamlit App for a **conditional Pokémon-style original creature generator**.

The user enters:

- One or two elemental types
- Six stats: HP, Attack, Defense, Special Attack, Special Defense, Speed
- Appearance description

The system outputs:

- Original creature image
- Name
- Type labels
- Current evolution stage
- Stats
- Pokédex-style description
- Evolution / devolution actions
- Debug information including LLM prompt, SDXL prompt, seed, and LoRA status

The project must satisfy the course requirement by combining:

- LLM: Gemini 2.5 Flash with Groq fallback
- Diffusion: SDXL
- Fine-tuning: LoRA
- Dataset: Kaggle Pokémon image datasets + PokeAPI metadata
- App: Streamlit
- Agent workflow: documented in `workflow_log.md`

---

## System Architecture

```text
User Input
  │
  ▼
Streamlit UI
  │
  ├── Type selectors
  ├── Stat sliders
  └── Appearance text area
  │
  ▼
Input Normalizer
  │
  ├── Validate type count <= 2
  ├── Validate stat ranges
  └── Compute Base Stat Total
  │
  ▼
Evolution Stage Estimator
  │
  ├── basic
  ├── stage_1
  └── stage_2
  │
  ▼
LLM Planner
  │
  ├── Gemini 2.5 Flash primary
  ├── Groq fallback
  ├── JSON schema validation
  └── Prompt repair if invalid JSON
  │
  ▼
Prompt Builder
  │
  ├── SDXL positive prompt
  ├── SDXL negative prompt
  ├── Type-specific visual tokens
  ├── Stat-specific visual traits
  └── Evolution lineage motifs
  │
  ▼
SDXL Generator
  │
  ├── Base SDXL
  ├── Optional LoRA weights
  ├── Seeded generation
  └── Save image
  │
  ▼
Lineage Store
  │
  ├── Save generated creature metadata
  ├── Save prompt and seed
  ├── Save parent-child relation
  └── Enable evolve / devolve
  │
  ▼
Streamlit Result Display
```

---

## Milestone 0 — Project Initialization

### Goal

Create a clean repository structure that Agents can extend safely.

### Tasks

- Create repo.
- Add `AGENTS.md`.
- Add `plan.md`.
- Add `.gitignore`.
- Add `.env.example`.
- Add empty `README.md`.
- Add empty `workflow_log.md`.
- Add base folders:
  - `src/`
  - `scripts/`
  - `configs/`
  - `data/`
  - `outputs/`
  - `tests/`

### `.gitignore` requirements

Include:

```text
.env
__pycache__/
*.pyc
.venv/
venv/
data/raw/
data/processed/lora_images/
outputs/
*.safetensors
*.ckpt
*.pt
*.pth
.DS_Store
```

### Expected Output

A repository skeleton that can be installed and extended.

### Commit

```bash
git add .
git commit -m "chore: initialize project structure"
```

---

## Milestone 1 — Environment Setup

### Goal

Create a reproducible Python environment.

### Recommended Python Version

Use Python 3.11.

### Initial dependencies

Prepare `requirements.txt` with at least:

```text
streamlit
torch
torchvision
diffusers
transformers
accelerate
peft
safetensors
pydantic
python-dotenv
requests
pillow
numpy
pandas
matplotlib
kaggle
google-generativeai
groq
pytest
pyyaml
```

The exact versions may be pinned after the first successful run.

### Tasks

- Create `requirements.txt`.
- Create `src/utils/config.py`.
- Implement environment variable loading.
- Implement config loading from `configs/app.yaml`.
- Verify imports.

### Setup command

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Expected Output

The environment can import all required packages.

### Validation

```bash
python -m compileall src scripts app.py
```

### Commit

```bash
git add .
git commit -m "chore: add environment and dependency setup"
```

---

## Milestone 2 — Data Collection

### Goal

Prepare image and metadata sources for LoRA training and prompt planning.

### Data Sources

Use:

- Kaggle Pokémon image datasets for images
- PokeAPI for structured metadata

### Tasks

#### 2.1 Kaggle image dataset

- Add instructions in README for downloading Kaggle dataset.
- Store images under:

```text
data/raw/kaggle_pokemon_images/
```

- Do not commit raw images.

#### 2.2 PokeAPI fetcher

Create:

```text
scripts/fetch_pokeapi.py
src/data/fetch_pokeapi.py
```

The fetcher should:

- Query Pokémon names
- Query types
- Query base stats
- Optionally query height, weight, abilities, and evolution chain
- Save raw JSON under:

```text
data/raw/pokeapi/
```

- Save normalized metadata under:

```text
data/processed/metadata.json
```

Suggested normalized format:

```json
{
  "id": 6,
  "name": "charizard",
  "types": ["fire", "flying"],
  "stats": {
    "hp": 78,
    "attack": 84,
    "defense": 78,
    "special_attack": 109,
    "special_defense": 85,
    "speed": 100
  },
  "base_stat_total": 534,
  "height": 17,
  "weight": 905,
  "abilities": ["blaze", "solar-power"]
}
```

### Expected Output

- `data/processed/metadata.json`
- small sample file that can be committed, such as:

```text
data/samples/pokeapi_sample.json
```

### Validation

```bash
python scripts/fetch_pokeapi.py --limit 50
```

### Commit

```bash
git add .
git commit -m "feat: add PokeAPI metadata fetcher"
```

---

## Milestone 3 — Caption Dataset Preparation

### Goal

Build a LoRA training dataset with image-caption pairs.

### Tasks

Create:

```text
scripts/prepare_lora_dataset.py
src/data/prepare_dataset.py
src/data/caption_builder.py
```

The script should:

1. Read Kaggle image paths.
2. Match images to Pokémon names when possible.
3. Join with PokeAPI metadata.
4. Generate generic captions.
5. Copy or resize selected images into:

```text
data/processed/lora_images/
```

6. Save captions to:

```text
data/processed/captions.jsonl
```

### Caption Rules

Use a generic style token:

```text
pokecreature_style
```

Caption format:

```text
pokecreature_style, an original {type_1} and {type_2} creature, {body_hint}, {color_hint}, cute monster illustration, clean line art, game creature concept art
```

If only one type:

```text
pokecreature_style, an original {type_1} creature, {body_hint}, {color_hint}, cute monster illustration, clean line art, game creature concept art
```

Avoid official names in captions when possible.

### Type-to-Visual Hints

Create a mapping file in:

```text
configs/prompt_templates.yaml
```

Example:

```yaml
type_visual_hints:
  fire:
    colors: ["red", "orange", "yellow"]
    motifs: ["flames", "warm glow", "ember particles"]
  water:
    colors: ["blue", "cyan", "white"]
    motifs: ["fins", "water droplets", "smooth aquatic body"]
  grass:
    colors: ["green", "yellow-green", "brown"]
    motifs: ["leaves", "vines", "flower buds"]
```

### Expected Output

- `data/processed/captions.jsonl`
- `data/samples/captions_sample.jsonl`
- A documented dataset preparation command

### Validation

```bash
python scripts/prepare_lora_dataset.py --max-images 500
```

### Commit

```bash
git add .
git commit -m "feat: add LoRA dataset preparation pipeline"
```

---

## Milestone 4 — Evolution Stage Estimator

### Goal

Classify a creature into an evolution stage using deterministic stat rules.

### Tasks

Create:

```text
src/evolution/stage_estimator.py
tests/test_stage_estimator.py
```

Initial rule:

```text
BST = HP + Attack + Defense + Special Attack + Special Defense + Speed
```

Stage thresholds:

```text
basic: BST <= 340
stage_1: 341 <= BST <= 480
stage_2: BST > 480
```

The function should return:

```python
{
    "base_stat_total": 420,
    "stage": "stage_1",
    "reason": "BST is between 341 and 480."
}
```

### Edge Cases

Test:

- Very low stats
- Boundary at 340
- Boundary at 341
- Boundary at 480
- Boundary at 481
- Invalid negative stats
- Invalid missing stat

### Expected Output

A deterministic stage estimator used before LLM planning.

### Validation

```bash
pytest tests/test_stage_estimator.py
```

### Commit

```bash
git add .
git commit -m "feat: implement evolution stage estimator"
```

---

## Milestone 5 — LLM Client with Gemini and Groq Fallback

### Goal

Implement a robust LLM planner client.

### Files

Create:

```text
src/llm/client.py
src/llm/gemini_client.py
src/llm/groq_client.py
src/llm/schemas.py
src/llm/planner.py
tests/test_llm_schema.py
```

### Required Behavior

The client should:

1. Try Gemini 2.5 Flash first.
2. If Gemini fails, try Groq fallback.
3. Return both result and provider metadata.
4. Validate JSON.
5. Retry JSON repair once if invalid.
6. If all attempts fail, return a structured error.

### LLM Output Schema

Use Pydantic.

Required fields:

```python
class CreaturePlan(BaseModel):
    name: str
    types: list[str]
    evolution_stage: Literal["basic", "stage_1", "stage_2"]
    visual_concept: str
    stat_interpretation: dict[str, str]
    core_motifs: list[str]
    color_palette: list[str]
    sdxl_prompt: str
    negative_prompt: str
    pokedex_entry: str
    evolution_hint: str
    devolution_hint: str
```

### Prompt Requirements

System prompt should tell the LLM:

- Generate an original creature, not an official Pokémon.
- Preserve user types and stats.
- Convert stats into visual features.
- Return JSON only.
- Avoid copyrighted character names.
- Keep the creature visually suitable for SDXL.

### Fallback Example

```text
Gemini failure: 503 Service Unavailable
Groq fallback: success
Provider used: groq
```

### Expected Output

A function such as:

```python
plan_creature(user_input: CreatureInput) -> PlannedCreatureResult
```

### Validation

```bash
pytest tests/test_llm_schema.py
```

### Commit

```bash
git add .
git commit -m "feat: implement Gemini to Groq fallback planner"
```

---

## Milestone 6 — Prompt Builder

### Goal

Convert validated LLM plan and deterministic rules into final SDXL prompts.

### Files

Create:

```text
src/generation/prompt_builder.py
tests/test_prompt_builder.py
```

### Prompt Requirements

The final prompt should include:

- `pokecreature_style` if LoRA is enabled
- Original creature wording
- Type motifs
- Stat-derived visual traits
- User appearance description
- Clean creature concept art style
- Single creature
- Full body
- Plain or simple background

Example:

```text
pokecreature_style, original fire and flying type creature, full body, cute monster concept art, orange and crimson color palette, flame-shaped feathers, aerodynamic wings, sharp eyes, fast agile body, clean line art, high quality illustration, simple background
```

Negative prompt:

```text
official pokemon, pikachu, charizard, copyrighted character, text, watermark, logo, blurry, low quality, malformed limbs, extra heads, duplicate creature, cropped body
```

### Expected Output

A deterministic prompt builder that can work even if the LLM returns only partial information.

### Validation

```bash
pytest tests/test_prompt_builder.py
```

### Commit

```bash
git add .
git commit -m "feat: add SDXL prompt builder"
```

---

## Milestone 7 — Base SDXL Generation

### Goal

Make image generation work before adding LoRA.

### Files

Create:

```text
src/generation/sdxl_pipeline.py
scripts/generate_sample.py
```

### Requirements

The SDXL generator should:

- Load SDXL base model.
- Accept prompt and negative prompt.
- Accept seed.
- Accept inference steps.
- Accept guidance scale.
- Save output image.
- Return image path.

### Suggested Config

Create:

```text
configs/app.yaml
```

Example:

```yaml
generation:
  model_id: "stabilityai/stable-diffusion-xl-base-1.0"
  width: 1024
  height: 1024
  num_inference_steps: 30
  guidance_scale: 7.0
  use_lora: false
  lora_path: ""
```

### GPU Memory Notes

If GPU memory is limited:

- Enable fp16.
- Enable attention slicing.
- Enable VAE slicing.
- Reduce resolution to 768 or 512 for testing.
- Reduce inference steps.

### Expected Output

A sample generated image under:

```text
outputs/generations/
```

### Validation

```bash
python scripts/generate_sample.py
```

### Commit

```bash
git add .
git commit -m "feat: add base SDXL generation pipeline"
```

---

## Milestone 8 — LoRA Training

### Goal

Fine-tune SDXL with LoRA on Pokémon-style creature images.

### Files

Create or adapt:

```text
scripts/train_lora_sdxl.py
configs/lora_sdxl.yaml
```

### Training Config

Example:

```yaml
pretrained_model_name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
train_data_dir: "data/processed/lora_images"
caption_column: "text"
resolution: 1024
train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
max_train_steps: 1000
checkpointing_steps: 250
rank: 16
mixed_precision: "fp16"
output_dir: "outputs/lora/pokecreature_sdxl_lora"
```

If hardware is limited, use:

```yaml
resolution: 768
max_train_steps: 500
rank: 8
```

### Training Command

Example:

```bash
accelerate launch scripts/train_lora_sdxl.py --config configs/lora_sdxl.yaml
```

If using the official Diffusers training script, document the exact command in README.

### Expected Output

LoRA weights under:

```text
outputs/lora/pokecreature_sdxl_lora/
```

### Quality Check

Generate 5 test images:

- fire type high attack
- water type high special defense
- grass type basic creature
- electric type high speed
- dragon type stage 2

Save under:

```text
outputs/demo/lora_tests/
```

### Commit

Do not commit the large LoRA weights unless explicitly allowed.

Commit the config and scripts:

```bash
git add scripts/train_lora_sdxl.py configs/lora_sdxl.yaml README.md
git commit -m "feat: add SDXL LoRA training pipeline"
```

---

## Milestone 9 — LoRA Inference Integration

### Goal

Allow the app and scripts to load LoRA weights.

### Files

Create:

```text
src/generation/lora_loader.py
```

Modify:

```text
src/generation/sdxl_pipeline.py
configs/app.yaml
```

### Requirements

- If `use_lora: true`, load the LoRA path.
- If LoRA path does not exist, show warning and use base SDXL.
- The UI should display LoRA status.
- Prompt should include `pokecreature_style` when LoRA is enabled.

### Validation

```bash
python scripts/generate_sample.py --use-lora
```

### Commit

```bash
git add .
git commit -m "feat: integrate LoRA inference"
```

---

## Milestone 10 — Lineage Store

### Goal

Save each generated creature and allow evolution/devolution based on previous metadata.

### Files

Create:

```text
src/evolution/lineage_store.py
tests/test_lineage_store.py
```

### Saved JSON Format

Save one file per lineage:

```text
outputs/lineages/{lineage_id}.json
```

Example:

```json
{
  "lineage_id": "ln_20260620_001",
  "creatures": [
    {
      "creature_id": "cr_001",
      "parent_id": null,
      "stage": "basic",
      "name": "Flameling",
      "types": ["fire"],
      "stats": {
        "hp": 45,
        "attack": 52,
        "defense": 43,
        "special_attack": 60,
        "special_defense": 50,
        "speed": 65
      },
      "core_motifs": ["ember tail", "round fox-like body"],
      "color_palette": ["orange", "cream", "red"],
      "prompt": "...",
      "negative_prompt": "...",
      "seed": 12345,
      "image_path": "outputs/generations/cr_001.png"
    }
  ]
}
```

### Expected Output

Each generated creature has persistent metadata.

### Validation

```bash
pytest tests/test_lineage_store.py
```

### Commit

```bash
git add .
git commit -m "feat: add creature lineage storage"
```

---

## Milestone 11 — Evolution and Devolution Planner

### Goal

Generate planned next or previous forms.

### Files

Create:

```text
src/evolution/evolution_planner.py
tests/test_evolution_planner.py
```

### Evolution Rules

If current stage is `basic`:

- Next stage becomes `stage_1`
- Increase total stats by about 20%
- Increase visual complexity
- Preserve core motifs
- Add one stronger type motif
- Keep the same type unless the user explicitly allows type change

If current stage is `stage_1`:

- Next stage becomes `stage_2`
- Increase total stats by about 15%
- Make form more powerful, confident, and elaborate

If current stage is `stage_2`:

- Do not evolve further by default
- UI should show disabled evolve button

### Devolution Rules

If current stage is `stage_2`:

- Previous stage becomes `stage_1`
- Reduce total stats by about 15%
- Simplify silhouette
- Keep lineage motifs

If current stage is `stage_1`:

- Previous stage becomes `basic`
- Reduce total stats by about 20%
- Make the creature smaller, softer, and younger

If current stage is `basic`:

- Do not devolve
- UI should show disabled devolve button

### LLM Role

The deterministic evolution planner should create transformed stats and required lineage constraints.

The LLM should then generate:

- evolved / devolved name
- visual concept
- updated SDXL prompt
- updated Pokédex-style description

The LLM must preserve:

- lineage identity
- core motifs
- color palette
- type identity

### Expected Output

Functions:

```python
plan_evolution(creature_metadata) -> CreatureInput
plan_devolution(creature_metadata) -> CreatureInput
```

### Validation

```bash
pytest tests/test_evolution_planner.py
```

### Commit

```bash
git add .
git commit -m "feat: implement evolution and devolution planning"
```

---

## Milestone 12 — Streamlit MVP

### Goal

Build the working UI.

### File

Create:

```text
app.py
src/ui/components.py
src/ui/state.py
```

### Required UI

#### Sidebar

- LLM primary provider
- Fallback provider
- SDXL model ID
- LoRA enabled toggle
- LoRA path
- Seed
- Inference steps
- Guidance scale

#### Main Form

- Type 1 dropdown
- Type 2 dropdown, optional
- HP slider
- Attack slider
- Defense slider
- Special Attack slider
- Special Defense slider
- Speed slider
- Appearance description text area
- Generate button

#### Result Panel

- Image
- Name
- Types
- Evolution stage
- Stats
- Base Stat Total
- Pokédex-style description
- Evolution hint
- Devolution hint

#### Controls

- Evolve button
- Devolve button
- Save result button, optional

#### Debug Expander

- Provider used
- Raw LLM JSON
- Final prompt
- Negative prompt
- Seed
- LoRA status
- Image path

### Expected Output

The app runs with:

```bash
streamlit run app.py
```

### Manual Test

Use input:

```text
Types: fire, flying
Stats: HP 70, Attack 95, Defense 70, Special Attack 110, Special Defense 75, Speed 120
Appearance: a small dragon-like creature with feathered wings and a glowing flame crest
```

Expected:

- Stage likely `stage_2`
- Fire/flying visual motifs
- High speed and special attack reflected in prompt
- Generated image appears
- Devolve button enabled
- Evolve button disabled or marked as unavailable

### Commit

```bash
git add .
git commit -m "feat: build Streamlit MVP"
```

---

## Milestone 13 — Evaluation and Demo Samples

### Goal

Produce enough examples to prove the system works.

### Files

Create:

```text
scripts/evaluate_outputs.py
outputs/demo/
```

### Test Cases

Generate at least 6 examples:

1. Basic fire creature
2. Stage 1 water / ice creature
3. Stage 2 dragon / electric creature
4. High defense rock creature
5. High speed flying creature
6. Evolution chain with basic → stage 1 → stage 2
7. Devolution example with stage 2 → stage 1 or stage 1 → basic

### Evaluation Criteria

Record manually:

- Type consistency
- Stat-to-visual consistency
- Prompt quality
- Image quality
- Evolution consistency
- App usability

A simple table is enough:

```markdown
| Case | Type consistency | Stat consistency | Image quality | Notes |
|---|---:|---:|---:|---|
| Fire/Flying high speed | 4/5 | 4/5 | 4/5 | Wings and flame crest visible |
```

### Expected Output

- Demo screenshots
- Demo generation images
- Small evaluation table in README or workflow log

### Commit

```bash
git add README.md workflow_log.md outputs/demo/
git commit -m "docs: add demo samples and evaluation notes"
```

If output files are too large, commit only selected compressed screenshots.

---

## Milestone 14 — README.md

### Goal

Create the final project documentation.

### Required Sections

```markdown
# Conditional Pokémon-style Creature Generator

## Overview

## Features

## System Architecture

## Tech Stack

## Dataset

## Model Pipeline

## LLM Planner

## SDXL + LoRA

## Evolution and Devolution

## Installation

## Environment Variables

## Data Preparation

## LoRA Training

## Running the App

## Example Usage

## Agent Workflow Summary

## Limitations

## Future Work
```

### Important README Notes

Include:

- This is an academic project.
- It generates original Pokémon-style creatures.
- Raw datasets are not included in the repo.
- Users must download Kaggle data themselves.
- API keys are required for Gemini and Groq.
- LoRA weights may not be included if too large.

### Commit

```bash
git add README.md
git commit -m "docs: complete README"
```

---

## Milestone 15 — Workflow Log

### Goal

Complete required Agent collaboration record.

### File

Use:

```text
workflow_log.md
```

### Required Entries

At minimum, record:

1. Initial project proposal
2. Architecture planning
3. Dataset pipeline design
4. LLM fallback implementation
5. SDXL generation implementation
6. LoRA training setup
7. Evolution/devolution design
8. Streamlit UI implementation
9. Debugging notes
10. Final demo preparation

### Entry Format

```markdown
## 2026-06-20 - LLM Planner Design

### Goal
Design a structured planner that converts user stats and type inputs into SDXL prompts.

### Tool / Model
Codex CLI / Gemini / ChatGPT / other agent

### Prompt
...

### Output Summary
...

### Human Decision
...

### Issue / Fix
...

### Commit
feat: implement Gemini to Groq fallback planner
```

### Commit

```bash
git add workflow_log.md
git commit -m "docs: complete agent workflow log"
```

---

## Milestone 16 — Final Demo Material

### Goal

Prepare the submitted demonstration file.

### Required Demonstration

The video or screenshots should show:

1. App launch
2. User enters types, stats, and appearance description
3. App generates a creature
4. App displays evolution stage
5. App evolves a basic or stage 1 creature
6. App devolves a stage 1 or stage 2 creature
7. Result metadata and image are visible

### Recommended Demo Flow

Use three scenes:

#### Scene 1 — Basic Generation

Input:

```text
Type: grass
Stats: low to medium
Appearance: small turtle-like creature with leaf shell
```

Show:

- Basic stage
- Generated image
- Evolve button

#### Scene 2 — Evolution

Click evolve.

Show:

- Stage 1 or stage 2 generated form
- Same motif preserved
- Stronger visual design

#### Scene 3 — High Stat Devolution

Input:

```text
Type: dragon, electric
Stats: high values
Appearance: fast thunder dragon with crystal horns
```

Show:

- Stage 2
- Devolve button
- Devolved form

### Output File

Submit as:

```text
<StudentID>_HW7.mp4
```

or:

```text
<StudentID>_HW7.png
```

Repository link must be placed in:

```text
<StudentID>_HW7.txt
```

---

## Milestone 17 — Final Checklist

Before submission, verify:

### Code

- [ ] `streamlit run app.py` works.
- [ ] Gemini primary path works.
- [ ] Groq fallback path works or is documented.
- [ ] SDXL generation works.
- [ ] LoRA loading works or fallback is documented.
- [ ] Evolution works.
- [ ] Devolution works.
- [ ] Outputs are saved.

### Repository

- [ ] Source code is committed.
- [ ] README.md is complete.
- [ ] workflow_log.md is complete.
- [ ] requirements.txt exists.
- [ ] `.env.example` exists.
- [ ] `.env` is not committed.
- [ ] Raw dataset is not committed.
- [ ] Large model weights are not accidentally committed.

### Demo

- [ ] Demo video or screenshot exists.
- [ ] Demo shows core features.
- [ ] GitHub repo is public.
- [ ] `<StudentID>_HW7.txt` contains repo link.
- [ ] `<StudentID>_HW7.<ext>` is prepared.

---

## Suggested Development Order

Follow this order strictly:

1. Repo scaffold
2. Environment
3. PokeAPI metadata
4. Dataset caption preparation
5. Stage estimator
6. LLM planner
7. Prompt builder
8. Base SDXL generation
9. Streamlit MVP without LoRA
10. Evolution / devolution
11. LoRA training
12. LoRA inference
13. Demo samples
14. README
15. Workflow log
16. Final submission material

Reason:

The app should work even before LoRA is finished. LoRA improves style quality, but the complete project should not depend on LoRA training succeeding at the last minute.

---

## Risk Management

### Risk 1 — Gemini API 503 or quota issue

Mitigation:

- Implement Groq fallback.
- Cache successful LLM plans.
- Add deterministic fallback prompt builder.

### Risk 2 — SDXL GPU memory issue

Mitigation:

- Lower resolution for testing.
- Use fp16.
- Enable attention slicing.
- Reduce inference steps.
- Use CPU only as last resort for very slow demo.

### Risk 3 — LoRA training quality is unstable

Mitigation:

- Keep base SDXL generation working.
- Use a small, clean dataset subset.
- Use generic captions.
- Save multiple checkpoints.
- Pick the best checkpoint manually for demo.

### Risk 4 — Generated creature looks too much like official Pokémon

Mitigation:

- Use original creature wording.
- Avoid official character names in prompts.
- Add negative prompts for official names.
- Keep outputs as academic fan-style original concepts.

### Risk 5 — Evolution chain is visually inconsistent

Mitigation:

- Store core motifs and color palette.
- Use previous metadata in evolution prompt.
- Use deterministic seed offsets.
- Generate multiple candidates and select the best manually for demo if necessary.

---

## Definition of Done

The project is done when:

- A user can run `streamlit run app.py`.
- A user can enter type, stats, and description.
- The app generates an image and metadata.
- The app displays evolution stage.
- The user can evolve or devolve a generated creature when allowed.
- The repo contains README, workflow log, source code, and setup instructions.
- The demo material proves the app actually runs.
