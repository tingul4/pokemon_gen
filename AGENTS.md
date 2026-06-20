# AGENTS.md

## Project Identity

This repository implements a course project for **Deep Generative Models**.

Project name:

**Conditional Pokémon-style Creature Generator with LLM + SDXL LoRA**

Chinese name:

**條件式寶可風格原創怪獸生成器**

The system generates an original creature image and metadata from structured user inputs:

- Type 1 and optional Type 2
- HP, Speed, Special Attack, Special Defense, Attack, Defense
- Appearance description
- Optional evolution / devolution action based on the generated creature lineage

The project must be implemented as a working **Streamlit App** with source code, README, workflow log, and demonstration material.

This project is for academic demonstration. The system should be described as generating **original Pokémon-style creatures**, not official Pokémon characters. Do not claim affiliation with Nintendo, Game Freak, Creatures Inc., or The Pokémon Company.

---

## Core Technical Route

Agents must preserve the following technical route unless the human developer explicitly changes it.

### LLM

Primary LLM:

- `gemini-2.5-flash`

Fallback LLM provider:

- Groq free-tier model
- The exact Groq model name must be configurable through environment variables.

Required behavior:

- The LLM must not directly generate final images.
- The LLM is responsible for planning, prompt expansion, evolution logic, metadata generation, and output validation.
- The LLM client must support automatic fallback:
  1. Try Gemini first.
  2. If Gemini fails due to timeout, 503, quota, rate limit, or invalid response, retry with Groq.
  3. If both fail, return a structured error to the Streamlit UI.

### Image Generation

Base model:

- SDXL

Implementation framework:

- Hugging Face `diffusers`
- PyTorch
- `accelerate`
- `peft` / LoRA support where applicable

Fine-tuning method:

- LoRA

Dataset route:

- Kaggle Pokémon image datasets for visual style fine-tuning
- PokeAPI for structured metadata, including type, base stats, names, and evolution-related metadata where useful

Frontend:

- Streamlit

Do not replace Streamlit with Gradio unless explicitly instructed.

---

## Scope Control

The project must prioritize a stable MVP over ambitious features.

### Required MVP

The MVP must include:

1. Streamlit input form
   - Type 1
   - Type 2, optional
   - HP
   - Speed
   - Special Attack
   - Special Defense
   - Attack
   - Defense
   - Appearance description

2. LLM planner
   - Converts structured inputs into a creature concept
   - Generates SDXL prompt and negative prompt
   - Generates name, evolution stage, type explanation, stat interpretation, and Pokédex-style description

3. SDXL image generation
   - Runs base SDXL first
   - Supports optional LoRA loading after LoRA training is ready

4. LoRA fine-tuning pipeline
   - Dataset preparation script
   - Caption generation / metadata mapping
   - LoRA training script or documented training command
   - Inference script that loads the trained LoRA

5. Evolution / devolution
   - Generated creature must display its current evolution stage.
   - If the creature is basic or stage 1, the app can generate a planned evolved form.
   - If the creature is stage 1 or stage 2, the app can generate a simplified previous form.
   - Evolution and devolution must preserve core lineage motifs.

6. Output panel
   - Generated image
   - Name
   - Types
   - Stats
   - Evolution stage
   - Pokédex-style description
   - Prompt details, optionally hidden under an expander

7. README.md
   - Installation
   - API key setup
   - Dataset preparation
   - LoRA training
   - Streamlit execution
   - Known limitations

8. Workflow log
   - Records agent prompts, tools used, implementation decisions, bugs, fixes, and final results

### Optional Features

Only implement optional features after the MVP works end-to-end.

Optional features:

- Batch generation
- Shiny / alternate color variant
- Radar chart for stats
- Save / load generated creature lineage
- Gallery page
- Prompt comparison between Gemini and Groq
- ControlNet or IP-Adapter
- Better visual consistency for evolution chain
- Export creature card as PNG

Do not start optional features before the MVP image generation path works.

---

## Repository Structure

Use the following structure unless there is a strong reason to change it.

```text
.
├── AGENTS.md
├── plan.md
├── README.md
├── workflow_log.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
├── configs/
│   ├── app.yaml
│   ├── lora_sdxl.yaml
│   └── prompt_templates.yaml
├── data/
│   ├── raw/
│   │   ├── kaggle_pokemon_images/
│   │   └── pokeapi/
│   ├── processed/
│   │   ├── lora_images/
│   │   ├── captions.jsonl
│   │   └── metadata.json
│   └── samples/
├── outputs/
│   ├── generations/
│   ├── lineages/
│   └── demo/
├── src/
│   ├── __init__.py
│   ├── llm/
│   │   ├── client.py
│   │   ├── gemini_client.py
│   │   ├── groq_client.py
│   │   ├── planner.py
│   │   └── schemas.py
│   ├── generation/
│   │   ├── sdxl_pipeline.py
│   │   ├── lora_loader.py
│   │   └── prompt_builder.py
│   ├── data/
│   │   ├── fetch_pokeapi.py
│   │   ├── prepare_dataset.py
│   │   └── caption_builder.py
│   ├── evolution/
│   │   ├── stage_estimator.py
│   │   ├── evolution_planner.py
│   │   └── lineage_store.py
│   ├── ui/
│   │   ├── components.py
│   │   └── state.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── image_io.py
├── scripts/
│   ├── fetch_pokeapi.py
│   ├── prepare_lora_dataset.py
│   ├── train_lora_sdxl.py
│   ├── generate_sample.py
│   └── evaluate_outputs.py
└── tests/
    ├── test_stage_estimator.py
    ├── test_prompt_builder.py
    ├── test_llm_schema.py
    └── test_lineage_store.py
```

---

---

## Environment Management with uv

This project must use **uv venv** for Python environment management.

Do not use Conda as the default environment workflow unless the human developer explicitly requests it.

### Required Python Version

Use Python 3.11.

Recommended setup:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
uv venv .venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### Dependency File

`requirements.txt` is required and must be kept in the repository root.

Agents must update `requirements.txt` whenever a new runtime dependency is introduced.

The project should not depend on undocumented manual package installation.

### Dependency Installation Rule

Use:

```bash
uv pip install -r requirements.txt
```

Do not use plain `pip install` in project instructions unless documenting a fallback for users who do not have uv installed.

### Adding New Packages

When adding a package, use:

```bash
uv pip install <package-name>
```

Then update `requirements.txt`.

If exact package versions are known to work, pin them. If not, keep version constraints minimal during early development and pin versions after the MVP runs successfully.

Recommended after the first stable run:

```bash
uv pip freeze > requirements-lock.txt
```

`requirements-lock.txt` is optional, but useful for reproducibility.

### Running Project Commands

After activating `.venv`, run commands normally:

```bash
streamlit run app.py
python scripts/fetch_pokeapi.py
python scripts/prepare_lora_dataset.py
python scripts/generate_sample.py
pytest
```

Agents may also use `uv run` only if the project later adopts a `pyproject.toml` workflow. For the current project, the canonical workflow is:

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Environment Validation

Before committing environment-related changes, run:

```bash
python -m compileall src scripts app.py
python -c "import streamlit, torch, diffusers, transformers, accelerate, peft, pydantic"
```

When tests exist, also run:

```bash
pytest
```

### Environment Files

The repository should include:

```text
requirements.txt
.env.example
```

The repository must not commit:

```text
.venv/
.env
```

`.gitignore` must include both.

## Environment and Secrets

Use `.env` for secrets.

Required environment variables:

```bash
GEMINI_API_KEY=
GROQ_API_KEY=
GROQ_FALLBACK_MODEL=
HF_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
```

Rules:

- Never commit `.env`.
- Never hardcode API keys.
- Provide `.env.example`.
- The app must fail gracefully if keys are missing.
- The app must clearly show which LLM provider was used for a generation.

---

## Git and Version Control Rules

Use Git from the beginning.

### Branching

Default branch:

- `main`

Feature branches:

- `feature/data-pipeline`
- `feature/llm-planner`
- `feature/sdxl-generation`
- `feature/lora-training`
- `feature/evolution-system`
- `feature/streamlit-ui`
- `feature/docs-demo`

Small projects may work directly on `main`, but only if commits are clean and frequent.

### Commit Frequency

Commit after every meaningful unit of work.

Required commit points:

- Initial project scaffold
- Environment and dependency setup
- PokeAPI data fetch script
- Kaggle dataset preparation script
- Caption generation pipeline
- LLM client and fallback mechanism
- Prompt schema and validation
- SDXL base generation working
- LoRA loading working
- LoRA training command documented or implemented
- Evolution stage estimator implemented
- Evolution function implemented
- Devolution function implemented
- Streamlit MVP completed
- README completed
- Workflow log updated
- Demo outputs added

Avoid one large final commit.

### Commit Message Format

Use concise conventional-style messages:

```text
feat: add PokeAPI metadata fetcher
feat: implement Gemini to Groq fallback client
feat: add SDXL generation pipeline
feat: implement evolution stage estimator
fix: handle invalid LLM JSON response
docs: add LoRA training instructions
chore: update requirements
```

### Before Each Commit

Run at least:

```bash
python -m compileall src scripts app.py
```

When tests exist, also run:

```bash
pytest
```

Do not commit broken code unless the commit is explicitly marked as a WIP checkpoint and immediately followed by a fixing commit.

---

## Data Rules

### Raw Data

Raw Kaggle image files must stay under:

```text
data/raw/kaggle_pokemon_images/
```

Raw PokeAPI files must stay under:

```text
data/raw/pokeapi/
```

Do not commit large raw datasets to GitHub.

The `.gitignore` must exclude:

```text
data/raw/
data/processed/lora_images/
outputs/
*.safetensors
*.ckpt
*.pt
*.pth
.env
```

### Processed Data

Processed captions and metadata may be committed only if they are small and do not contain large copyrighted images.

Allowed examples:

```text
data/processed/captions_sample.jsonl
data/processed/metadata_schema.json
data/samples/sample_inputs.json
```

### Dataset Usage

Use Kaggle Pokémon image datasets for academic LoRA fine-tuning only.

Use PokeAPI to build structured metadata:

- type
- base stats
- possible evolution chains
- height / weight where useful
- abilities where useful

Do not build a model that directly reproduces official characters by name. Captions should prefer generic descriptors such as:

```text
a small fire-type creature with orange fur, flame-shaped tail, cute monster illustration style
```

Avoid captions such as:

```text
official Pikachu, official Charizard, exact Pokémon character
```

The target system should generate original creatures with inspired style, not replicas.

---

## LLM Planning Contract

The LLM planner must return structured JSON, not free-form prose.

Use a schema similar to:

```json
{
  "name": "string",
  "types": ["fire", "flying"],
  "evolution_stage": "basic | stage_1 | stage_2",
  "visual_concept": "string",
  "stat_interpretation": {
    "hp": "string",
    "attack": "string",
    "defense": "string",
    "special_attack": "string",
    "special_defense": "string",
    "speed": "string"
  },
  "core_motifs": ["string"],
  "color_palette": ["string"],
  "sdxl_prompt": "string",
  "negative_prompt": "string",
  "pokedex_entry": "string",
  "evolution_hint": "string",
  "devolution_hint": "string"
}
```

Rules:

- Validate all LLM outputs with Pydantic or equivalent schema validation.
- If the LLM returns invalid JSON, retry once with a repair prompt.
- If JSON repair fails, use a deterministic fallback prompt builder.
- Keep prompts safe and original.
- Do not request official copyrighted character replication.

---

## Stat-to-Visual Mapping Rules

Use stats as semantic conditioning, not as strict numerical diffusion conditioning.

Recommended mapping:

- High HP: bulky body, round silhouette, thick torso, resilient appearance
- High Attack: claws, horns, muscular limbs, sharp physical weapons
- High Defense: armor plates, shell, rocky skin, shield-like body parts
- High Special Attack: glowing organs, elemental aura, magical projectile motifs
- High Special Defense: mystical markings, protective aura, crystal or spiritual ornaments
- High Speed: lean body, aerodynamic silhouette, long legs, wings, dynamic pose

The LLM should convert stats into visual traits before calling SDXL.

---

## Evolution Stage Rules

Use a deterministic stage estimator before asking the LLM to write flavor text.

Recommended initial rule based on Base Stat Total:

```text
BST = HP + Attack + Defense + Special Attack + Special Defense + Speed
```

Initial thresholds:

- `basic`: BST <= 340
- `stage_1`: 341 <= BST <= 480
- `stage_2`: BST > 480

The thresholds can be adjusted after testing.

The stage estimator must be deterministic and unit-tested.

### Evolution

Evolution should:

- Increase visual complexity
- Preserve type identity
- Preserve core motifs
- Increase or redistribute stats
- Make the body larger, sharper, more confident, or more elemental
- Keep the generated creature in the same lineage

### Devolution

Devolution should:

- Simplify the silhouette
- Reduce ornamentation
- Lower or soften stats
- Preserve the same color family and core motifs
- Make the creature look younger or less developed

### Lineage Metadata

Each generation should save lineage metadata:

```json
{
  "lineage_id": "string",
  "creature_id": "string",
  "parent_id": "string | null",
  "stage": "basic | stage_1 | stage_2",
  "types": ["string"],
  "stats": {},
  "core_motifs": [],
  "color_palette": [],
  "prompt": "string",
  "negative_prompt": "string",
  "seed": 12345,
  "image_path": "string"
}
```

Evolution and devolution must use the previous lineage metadata, not only the image.

---

## SDXL and LoRA Rules

### Base Generation

First make the base SDXL path work without LoRA.

Minimum requirements:

- Load SDXL base model
- Generate one image from prompt
- Save image to `outputs/generations/`
- Return image path to Streamlit

### LoRA Integration

After base generation works:

- Add optional LoRA path in config
- Load LoRA weights only when path exists
- Display whether LoRA was used
- If LoRA loading fails, fall back to base SDXL and show warning

### LoRA Training

Use `diffusers` LoRA training flow or a project-local script based on it.

Training target:

- Learn Pokémon-like creature illustration style
- Avoid overfitting to exact official characters

Caption strategy:

- Use image metadata from PokeAPI when possible
- Use generic type and visual descriptors
- Include style token, such as `pokecreature_style`, only if consistently used in captions and inference

Example caption:

```text
pokecreature_style, an original electric-type creature, small body, yellow color palette, cute monster illustration, clean line art, game creature concept art
```

Do not train or caption with prompts designed to clone exact official characters.

---

## Streamlit App Rules

The Streamlit app must be the main demo entry point.

Run command:

```bash
streamlit run app.py
```

Required UI sections:

1. Sidebar configuration
   - LLM provider status
   - SDXL / LoRA settings
   - Seed
   - Number of inference steps
   - Guidance scale

2. Input form
   - Type selectors
   - Stat sliders
   - Appearance description text area

3. Generate button

4. Main result panel
   - Image
   - Name
   - Types
   - Evolution stage
   - Stats
   - Pokédex-style description

5. Evolution controls
   - Evolve button if current stage is `basic` or `stage_1`
   - Devolve button if current stage is `stage_1` or `stage_2`

6. Debug expander
   - LLM provider used
   - Raw LLM JSON
   - SDXL prompt
   - Negative prompt
   - Seed
   - LoRA path

The app should not crash on API failure or GPU out-of-memory. Show readable error messages.

---

## Code Quality Rules

Use clear modular Python.

Required practices:

- Use type hints where practical.
- Use dataclasses or Pydantic models for structured data.
- Keep API logic separate from UI.
- Keep image generation logic separate from prompt planning.
- Avoid hardcoded absolute paths.
- Use config files for model paths and generation parameters.
- Log important generation events.
- Write small tests for deterministic functions.

Do not put all code into `app.py`.

---

## Testing Rules

At minimum, test:

- Stat-to-stage mapping
- Prompt builder
- LLM JSON schema validation
- Gemini-to-Groq fallback behavior using mocked failures
- Lineage save/load
- Evolution and devolution stat transformation

Use:

```bash
pytest
```

If GPU tests are too slow, mark them as manual.

---

## Workflow Log Rules

Maintain `workflow_log.md`.

Each meaningful Agent interaction should record:

```markdown
## YYYY-MM-DD - Short Title

### Goal
What the agent was asked to do.

### Tool / Model
Which agent, CLI, or LLM was used.

### Prompt
The important prompt or summarized prompt.

### Output Summary
What the agent produced.

### Human Decision
What was accepted, rejected, or modified.

### Issue / Fix
Any bug, limitation, or debugging result.

### Commit
Git commit hash or commit message.
```

The workflow log is a required deliverable. Do not leave it until the end.

---

## README Requirements

README.md must include:

- Project overview
- System architecture
- Features
- Tech stack
- Dataset sources
- Setup instructions
- Environment variables
- How to fetch / prepare data
- How to train LoRA
- How to run Streamlit app
- Example inputs and outputs
- Limitations
- Agent workflow summary
- Demo material link or screenshot

---

## Demonstration Requirements

The final demo must show:

1. Basic generation from user inputs
2. Displayed evolution stage
3. Evolution generation
4. Devolution generation
5. At least one generated image using LoRA if LoRA training is completed
6. LLM fallback behavior, if feasible to demonstrate
7. Streamlit UI interaction

Save demo assets under:

```text
outputs/demo/
```

Do not commit large videos if the repository size becomes too large. Provide the submitted demo file separately if needed.

---

## Priority Order

When there is a trade-off, follow this order:

1. Stable uv-based environment setup
2. Working end-to-end Streamlit demo
3. Stable LLM prompt planning and JSON validation
4. Base SDXL generation
5. LoRA training and loading
6. Evolution / devolution consistency
7. Documentation and workflow log
8. Optional visual polish

A simple working system is better than an incomplete complex system.
