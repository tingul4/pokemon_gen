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

## 2026-06-21 - HLR Dataset Rebuild and Full LoRA Run

### Goal
Replace the previous fine-tuning data with Kaggle `hlrhegemony/pokemon-image-dataset`, remove old `data/` and `outputs/`, rebuild annotations with image labels that include PokeAPI metadata, then fine-tune LoRA on the full dataset and compare original SDXL against fused LoRA output.

### Tool / Model
Codex coding agent, Kaggle CLI, PokeAPI fetcher, CUDA GPU 0, SDXL base, diffusers LoRA training and fusion.

### Prompt
The human requested using the dataset organized by Pokemon name and multiple actions per Pokemon, matching names through PokeAPI to build annotation JSONL with image path, types, six stats, and appearance description. The old data and test outputs should be removed and replaced before full fine-tuning with longer steps.

### Output Summary
Removed old local `data/` and `outputs/`, downloaded the HLR Kaggle dataset to `data/raw/kaggle_pokemon_image_dataset/images/`, fetched 898 PokeAPI records, and prepared 2503 processed 512px images. The new `data/processed/annotations.jsonl` contains source image, processed image, source name, matched PokeAPI name, Pokemon id, types, six stats, base stat total, height, weight, abilities, generated appearance description, and compact LoRA caption. The compact `captions.jsonl` stays within SDXL CLIP token limits; the longest checked caption was 74 tokens.

### Human Decision
Use the full rebuilt dataset for the formal LoRA run. No fully idle GPU was available, so GPU 0 was selected because it had enough free memory.

### Issue / Fix
The first long training attempt showed SDXL caption truncation warnings because labels included long descriptions. The caption builder was changed to keep training captions compact while preserving the full label in `annotations.jsonl`. Dataset name aliases were added for special forms and encoded folder names such as `Nidoran`, `Mr. Mime`, `Porygon-Z`, `Flabebe`, Tapu names, and form-specific PokeAPI slugs. After rebuilding, all 2503 annotations matched metadata. Full LoRA training completed for 1500 steps at 512px, rank 8, fp32/no mixed precision, average loss `0.04094220210867934`. Visual comparison outputs were saved under `outputs/demo/hlr_dataset_comparison/`; `lora_scale=0.3` produced a better balance than `0.5`.

### Commit
feat: rebuild HLR dataset annotations

## 2026-06-21 - PokeAPI Species Descriptions

### Goal
Improve `annotation.jsonl` appearance descriptions because the previous rule-generated descriptions could drift from the source Pokemon images.

### Tool / Model
Codex coding agent, PokeAPI `pokemon` and `pokemon-species` endpoints, local dataset preparation scripts.

### Prompt
The human asked how appearance descriptions were generated, noted that they sometimes did not match the original Pokemon, and requested using more official descriptions from PokeAPI if available.

### Output Summary
The old appearance description was generated from type hints, highest stats, abilities, height, and weight. The fetcher now also calls each Pokemon's `pokemon-species` endpoint and stores `species_profile` with English genus, selected English official flavor text, flavor version, color, shape, habitat, egg groups, growth rate, and baby/legendary/mythical flags. `annotations.jsonl` now includes this official species profile and builds `appearance_description` from official PokeAPI text plus compact type/stat context.

### Human Decision
Use official PokeAPI species metadata in annotations, while keeping training captions compact and generic.

### Issue / Fix
Adding genus/color/shape to captions initially pushed 40 captions past SDXL's 77-token CLIP limit. The caption builder was shortened so all 2503 captions fit; max token length is now 70. The rebuilt annotations contain 2503 rows, 898 unique PokeAPI names, no missing metadata, no missing species profile, and no missing official flavor text.

### Commit
feat: use PokeAPI species descriptions

## 2026-06-21 - Remove Inferred Caption Traits

### Goal
Fix incorrect annotation summaries such as Abomasnow receiving `glowing elemental focus`, clarify whether LoRA uses captions as text input, and clean Unicode escape sequences from annotation files.

### Tool / Model
Codex coding agent, local data preparation scripts, pytest, SDXL CLIP tokenizer check.

### Prompt
The human observed that `training visual summary` could still be wrong because stat-derived phrases did not match the actual Pokemon appearance. They also asked whether LoRA uses captions as text input and whether inference converts frontend inputs to a similar format. Finally, they asked to clean escaped Unicode such as `\\u00e9` and `\\u2019` from annotations.

### Output Summary
Removed stat-to-visual trait inference from dataset captions and annotation appearance descriptions. Captions now use only `pokecreature_style`, type, compact official species profile terms such as genus/color/shape, numeric stats, and a short art-style phrase. Appearance descriptions now keep official PokeAPI profile text plus a neutral conditioning summary. The LoRA inference prompt builder now adds the same compact stats token format when LoRA is enabled. Processed metadata and annotations were normalized to ASCII-friendly text and written without JSON Unicode escapes.

### Human Decision
Keep official PokeAPI descriptions in annotations, but do not train on long official text or inferred visual traits.

### Issue / Fix
The root cause was `src/data/caption_builder.py` mapping high stats to visual labels and `ensure_ascii=True` writing escaped Unicode in annotation JSONL. Regenerated data removed `glowing elemental focus` and `strong claws or horns` from processed captions/annotations, removed `\\u` escapes, and kept all 2503 captions under the CLIP limit with max token length 55.

### Commit
fix: remove inferred visual traits from captions

## 2026-06-21 - Replace Dataset with Complete Pokedex

### Goal
Replace the fine-tuning dataset with `cristobalmitchell/pokedex`, remove PokeAPI from the annotation/caption pipeline, rebuild data from the dataset's images, descriptions, types, and six base stats, then retrain LoRA and compare a fused LoRA inference against base SDXL under the same input condition.

### Tool / Model
Codex coding agent, GitHub dataset clone, local Python CSV/image processing, CUDA GPU 0, SDXL base, diffusers LoRA training and fusion.

### Prompt
The human provided `https://github.com/cristobalmitchell/pokedex/tree/main` and requested using its images, descriptions, and stat fields directly. The old dataset should be replaced, PokeAPI could be removed, annotations and captions should come from the dataset itself, and the full dataset should be used for LoRA fine-tuning followed by merged/fused-weight inference comparison.

### Output Summary
Rebuilt `src/data/prepare_dataset.py` around the Complete Pokedex layout: UTF-16 tab-separated `data/pokemon.csv`, `images/large_images`, and `images/alt_images`. The processed dataset now contains 1137 image-caption pairs, 898 metadata records, 898 unique Pokemon names, no unmatched image files, no JSON Unicode escapes, and captions under the CLIP 77-token limit. The annotation JSONL includes source image, processed image, Pokemon name, national number, form, types, six stats, base stat total, height, weight, abilities, classification, official description, generation, flags, evolution chain, forms, appearance description, and compact LoRA caption.

### Human Decision
Use the Complete Pokedex dataset as the authoritative label source for fine-tuning annotations and captions, without requiring PokeAPI metadata matching.

### Issue / Fix
The dataset CSV is UTF-16 and tab-separated, not UTF-8 CSV, and `gen` is a Roman numeral string. The parser was adjusted accordingly. ASCII normalization initially collapsed `Nidoran♀` and `Nidoran♂` into the same name, so name cleaning now preserves them as `Nidoran Female` and `Nidoran Male`. LoRA training completed for 1500 steps at 512px/rank 8/fp32 with average loss `0.0632460796807427`. The fused comparison used `lora_scale=0.3` and saved `outputs/demo/pokedex_dataset_comparison/base_vs_fused_scale03/20260621_235736_cdd80921_comparison.png`; visually the fused image has cleaner game-creature line art and sharper form details than the base SDXL image.

### Commit
feat: switch LoRA data to Complete Pokedex dataset

## 2026-06-22 - LoRA Rank Alpha Sweep

### Goal
Try different LoRA `rank` and `lora_alpha` values on the Complete Pokedex dataset and compare their visual effects under the same prompt, seed, and inference settings.

### Tool / Model
Codex coding agent, CUDA GPU 1, SDXL base, diffusers LoRA training and fused-weight inference.

### Prompt
The human requested testing different LoRA rank or alpha values to see how the output quality changes.

### Output Summary
Updated `scripts/train_lora_sdxl.py` so `rank` and `lora_alpha` can be configured independently from the YAML file or CLI, and added both values plus trainable parameter count to `training_metrics.json`. Ran four controlled 800-step fp32 LoRA trainings at 512px on the same 1137-image processed dataset: `r4/a4`, `r8/a8`, `r8/a16`, and `r16/a16`. Average losses were `0.06394952945047408`, `0.06466321479703765`, `0.0644248302935739`, and `0.0643500281550223` respectively. Generated fixed-condition fused comparison sheets at `outputs/demo/lora_rank_alpha_sweep/20260622_104628/lora_rank_alpha_sweep.png` with `lora_scale=0.3` and `outputs/demo/lora_rank_alpha_sweep/20260622_104747_scale05/lora_rank_alpha_sweep_scale05.png` with `lora_scale=0.5`.

### Human Decision
Use a controlled sweep rather than replacing the previous 1500-step rank-8 production LoRA immediately, so visual differences can be inspected before selecting a new default.

### Issue / Fix
The previous training script hard-coded `lora_alpha` to equal `rank`, making alpha sweeps impossible. Adding `lora_alpha` to the config and CLI fixed this. Loss alone did not distinguish the variants clearly; visual comparison showed `r8/a16` was the most balanced for the fixed fire/flying prompt at `lora_scale=0.5`, while `r4/a4` and `r16/a16` pushed stronger phoenix-like fire-wing motifs.

### Commit
feat: add LoRA rank alpha sweep controls

## 2026-06-22 - Train Selected r16 a16 LoRA

### Goal
Use `rank=16` and `lora_alpha=16` as the selected LoRA hyperparameters, train for 4000 steps, and generate visual comparisons before deciding whether to connect the resulting weights to the Streamlit frontend.

### Tool / Model
Codex coding agent, CUDA GPU 1, SDXL base, diffusers LoRA training and fused-weight inference.

### Prompt
The human selected `r=16/a=16` and requested a 4000-step LoRA training run. If the visual effect looks good, the next step will be connecting it to the frontend.

### Output Summary
Trained `rank=16`, `lora_alpha=16` on the 1137-image Complete Pokedex processed dataset for 4000 steps at 512px/fp32. The run completed without OOM or non-finite loss, wrote weights to `outputs/lora/pokecreature_sdxl_lora_r16_a16_4000_20260622_1204/`, and produced average loss `0.06186473529139767` with `23224320` trainable parameters. Generated fixed-condition base-vs-fused comparisons with the previous fire/flying prompt at `lora_scale=0.3` and `0.5`.

### Human Decision
Do not connect the LoRA to the frontend yet; first inspect the generated comparison images.

### Issue / Fix
The `lora_scale=0.3` comparison preserved the intended small dragon-like creature form while improving line art, silhouette, and fire-wing styling. The `lora_scale=0.5` comparison pushed the creature toward a stronger phoenix-like form and showed more prompt drift. If this weight is connected to the frontend, `lora_scale=0.3` is the safer default.

### Commit
docs: record r16 a16 4000 step LoRA run

## 2026-06-22 - Connect r16 a16 LoRA to Frontend

### Goal
Connect the selected 4000-step `rank=16`, `lora_alpha=16` LoRA to the Streamlit frontend using `lora_scale=0.5`.

### Tool / Model
Codex coding agent, local Python code edits, Streamlit configuration, pytest.

### Prompt
The human requested using the `lora_scale=0.5` setting in the frontend.

### Output Summary
Updated `configs/app.yaml` so the Streamlit app enables LoRA by default, points to `outputs/lora/pokecreature_sdxl_lora_r16_a16_4000_20260622_1204`, and uses `lora_scale: 0.5`. Added a sidebar LoRA scale slider, stored the scale in generation metadata, and displayed it in the debug panel. Updated the SDXL generator and LoRA loader to fuse LoRA weights with the requested scale and to rebuild the pipeline when the LoRA path or scale changes. Added CLI demo support for `--lora-scale` and a unit test for LoRA fuse scale behavior.

### Human Decision
Use the stronger `lora_scale=0.5` setting as the frontend default despite the earlier visual note that it can push the fixed fire/flying prompt toward a stronger phoenix-like style.

### Issue / Fix
The previous app code only loaded LoRA weights and did not expose or apply a scale setting. The loader now calls `fuse_lora(lora_scale=...)` when available, and the generator tracks `path|scale` as the LoRA key to avoid stale fused weights when settings change.

### Commit
feat: enable r16 a16 LoRA in frontend

## 2026-06-22 - Fix Electric and Single-View Prompting

### Goal
Improve frontend and CLI inference prompts after manual testing showed electric creatures sometimes lacked visible lightning, and evolution/devolution could produce multi-view character-sheet compositions.

### Tool / Model
Codex coding agent, local Python code edits, pytest, compileall, SDXL LoRA smoke inference on CUDA GPU 0.

### Prompt
The human provided an electric stage-2 amphibian/turtle/frog example and reported two issues: the generated image lacked lightning elements, and evolved/devolved outputs sometimes showed multiple character views instead of one visual view.

### Output Summary
Updated SDXL prompt construction to prioritize single-view composition, elemental type motifs, user appearance text, and LoRA stats tokens within the CLIP token budget. Strengthened electric visual hints to include visible lightning bolts, electric arcs, and yellow sparks. Updated negative prompt merging so LLM-provided negative prompts are combined with project-level guards instead of replacing them, and added guards against multiple views, model sheets, turnarounds, front/side/back views, grids, and split screens. Applied the same negative prompt merging to the Streamlit app and CLI/demo scripts.

### Human Decision
Keep `lora_scale=0.5`; fix prompt conditioning rather than changing the selected LoRA weight.

### Issue / Fix
The root cause was prompt budget and negative-prompt replacement. The old positive prompt spent early budget on generic style, stats, and long descriptions, so electric motifs could be truncated. The old app also used the LLM negative prompt directly, losing fixed anti-character-sheet guards. The final electric test prompt is 75 CLIP tokens and the merged negative prompt is 73 CLIP tokens, both below the 77-token limit. A 20-step LoRA smoke image was generated at `outputs/demo/electric_prompt_fix/20260622_151712_41f8df83.png`; visual inspection showed one creature view with visible lightning arc and turtle/frog amphibian form.

### Commit
fix: improve electric and single-view prompting

## 2026-06-22 - Match LoRA Inference Captions

### Goal
Adjust LoRA inference input format to match `data/processed/captions.jsonl`, then test each elemental type with varied appearance descriptions to observe prompt adherence.

### Tool / Model
Codex coding agent, local Python edits, pytest, compileall, SDXL LoRA inference on CUDA GPU 0.

### Prompt
The human requested changing inference text to match the caption format used for LoRA training and trying every type with different appearance descriptions.

### Output Summary
Changed LoRA-enabled `build_sdxl_prompt` to emit caption-style prompts: `pokecreature_style, original {type}-type creature, {appearance descriptor}, single front view, stats hp... atk... def... spa... spd... spe..., clean game creature art`. Added `scripts/generate_type_sweep.py` to generate one LoRA sample per type, save the prompts and image paths in a summary JSON, and create a contact sheet for visual review. Updated README with the new inference format and sweep command.

### Human Decision
Keep LoRA scale at `0.5`, but align inference text with the training caption distribution.

### Issue / Fix
The first caption-style sweep at `outputs/demo/type_prompt_sweep_caption_format/20260622_153813_contact_sheet.png` improved type/style adherence but still produced sheet-like outputs for water, bug, rock, and dragon. Adding a short `single front view` token to the caption-style prompt plus stronger negative prompt terms improved the second sweep at `outputs/demo/type_prompt_sweep_caption_single_view/20260622_154107_contact_sheet.png`; water, bug, rock, and dragon became single main subjects. Poison still showed a possible small duplicate-like secondary form, so prompt adherence is improved but not perfect.

### Commit
feat: match LoRA inference caption format
