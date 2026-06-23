# Pokemon-style image generator

## 專題名稱

**Pokemon-style image generator**

中文名稱：**條件式寶可風格原創怪獸生成器**

本專題是 Deep Generative Models 課程專案，目標是讓使用者輸入屬性、六圍數值與外觀描述後，生成一張原創寶可風格怪獸圖片，並支援進化與退化。系統只生成原創怪獸，不宣稱與 Nintendo、Game Freak、Creatures Inc. 或 The Pokemon Company 有任何關聯。

## 功能摘要

- Streamlit 前端輸入表單：屬性 1、屬性 2、HP、攻擊、防禦、特攻、特防、速度、外觀描述。
- LLM 規劃器：使用 Gemini 2.5 Flash，失敗時自動 fallback 到 Groq。
- SDXL 圖像生成：使用 Hugging Face Diffusers 與 PyTorch。
- LoRA 權重：專案預設載入 `weights/` 底下已準備好的 LoRA，使用者不用先訓練即可啟動 app。
- LoRA 訓練復現：保留資料準備與訓練指令，作為需要重訓時的補充流程。
- IP-Adapter：進化與退化時預設使用前一階段圖片作為 reference，維持視覺一致性。
- 進化鏈暫存：前端會記錄 `basic / stage_1 / stage_2` 已生成圖片，回到已生成階段時直接復用原圖。
- 部署模式 UI：預設隱藏本地模型路徑、LoRA 路徑、圖片路徑與 raw debug JSON。

## 系統架構

使用者輸入會依序經過下列流程：

```text
Streamlit UI
  -> CreatureInput schema validation
  -> deterministic stage estimator
  -> LLM planner: Gemini 2.5 Flash, Groq fallback
  -> prompt builder
  -> SDXL base model
  -> optional LoRA weights
  -> optional IP-Adapter for evolution/devolution
  -> image output and lineage metadata
```

主要模組：

- `app.py`：Streamlit 入口，負責輸入、顯示、進化鏈控制與 deployment UI。
- `src/llm/`：Gemini、Groq、JSON schema 驗證、fallback planner。
- `src/generation/`：SDXL pipeline、LoRA 載入、IP-Adapter 載入、prompt builder。
- `src/evolution/`：進化階段估計、進化/退化規劃、lineage metadata 儲存。
- `src/data/`：資料集準備、caption/label 產生。
- `scripts/`：資料準備與 LoRA 訓練。
- `tests/`：確定性邏輯、prompt、schema、lineage、LoRA/IP-Adapter loader 測試。

## 使用技術

### LLM

- Primary：`gemini-2.5-flash`
- Fallback：Groq，模型名稱由 `GROQ_FALLBACK_MODEL` 設定
- 主要用途：
  - 將使用者輸入轉成怪獸概念
  - 產生名稱、屬性解釋、六圍解釋、圖鑑描述
  - 輔助進化與退化文字規劃
- LLM 不直接生成圖片，只負責 structured planning。

### Diffusion

- Base model：`stabilityai/stable-diffusion-xl-base-1.0`
- Framework：Hugging Face `diffusers`
- Runtime：PyTorch
- 主要流程：SDXL prompt conditioning -> optional LoRA -> optional IP-Adapter -> image output

### Adapter / Fine-tuning

- LoRA：使用 PEFT / Diffusers LoRA adapter，主要訓練 UNet attention layers。
- App 預設使用權重：

```text
weights/pokecreature_sdxl_lora_sks_anchor_r16_a16_20260623/
```

- LoRA 設定：
  - resolution：512
  - rank：16
  - lora_alpha：16
  - optimizer steps：3000
  - gradient_accumulation_steps：4
  - micro-batches：12000
  - average loss：`0.06041246440215036`

- IP-Adapter：
  - model id：`h94/IP-Adapter`
  - subfolder：`sdxl_models`
  - weight：`ip-adapter_sdxl.bin`
  - 前端進化與退化預設啟用，使用目前階段圖片作為 reference image。

## 資料集與來源

本專題使用 `cristobalmitchell/pokedex` 的圖片與 CSV metadata：

- GitHub：[cristobalmitchell/pokedex](https://github.com/cristobalmitchell/pokedex)
- Kaggle：[The Complete Pokedex Dataset](https://www.kaggle.com/cristobalmitchell/pokedex)

資料放置位置：

```bash
data/raw/cristobalmitchell_pokedex/
```

必要目錄結構：

```text
data/raw/cristobalmitchell_pokedex/
├── data/
│   └── pokemon.csv
└── images/
    └── large_images/
```

使用內容：

- 圖片：`data/raw/cristobalmitchell_pokedex/images/large_images/`
- 結構化資料：`data/raw/cristobalmitchell_pokedex/data/pokemon.csv`
- 欄位用途：
  - type
  - HP、Attack、Defense、Special Attack、Special Defense、Speed
  - classification
  - official website description

處理後資料：

- `data/processed/lora_images/`
- `data/processed/captions.jsonl`
- `data/processed/annotations.jsonl`
- `data/processed/metadata.json`
- `data/samples/captions_sample.jsonl`
- `data/samples/annotations_sample.jsonl`

目前處理後資料含 898 組 image-caption pairs。`lora_images` 直接複製 raw `large_images` 圖片，避免重新裁切或補背景造成破圖。

## LoRA Label 設計

訓練與 LoRA inference 使用一致的 label 形狀：

```text
sks style, single image, single creature, full body, blank background, clean composition, types electric, stats hp35 attack55 defense40 special_attack50 special_defense50 speed90, appearance yellow mouse-like creature with visible lightning bolts
```

設計原則：

- `sks style`：自訂 style token，用來綁定 LoRA 學到的寶可風格。
- `single image, single creature, full body, blank background, clean composition`：有語意的 anchor phrase，用來約束單張、單體、全身、乾淨構圖。
- `types ...`：屬性條件。
- `stats ...`：六圍條件。
- `appearance ...`：由使用者輸入或 CSV description 產生的外觀條件。

目前最小 negative prompt：

```text
multiple views, multi panel, grid layout, reference sheet
```

## 本地端直接使用

### 1. 建立 Python 環境

本專題使用 Python 3.11 與 uv：

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. 設定環境變數

複製 `.env.example`：

```bash
cp .env.example .env
```

填入：

```bash
GEMINI_API_KEY=
GROQ_API_KEY=
GROQ_FALLBACK_MODEL=
HF_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
```

直接使用 app 時只需要 LLM key 與 Hugging Face model access。`KAGGLE_USERNAME`、`KAGGLE_KEY` 只在重新準備資料或重訓 LoRA 時需要。

### 3. 確認 LoRA 權重

```bash
ls weights/pokecreature_sdxl_lora_sks_anchor_r16_a16_20260623/pytorch_lora_weights.safetensors
```

`configs/app.yaml` 已預設：

```yaml
generation:
  use_lora: true
  lora_path: weights/pokecreature_sdxl_lora_sks_anchor_r16_a16_20260623
```

因此啟動 app 後會直接載入專案內的 LoRA 權重。若權重檔不存在或載入失敗，app 會 fallback 到 base SDXL 並在進階資訊中顯示 LoRA 狀態。

### 4. 執行 Streamlit App

```bash
streamlit run app.py
```

預設服務網址：

```text
http://localhost:8501
```

## LoRA 訓練復現

一般使用者不需要執行本章節；只有要重現或重新訓練 LoRA 時才需要下載資料集與跑訓練。

`data/` 是可重新產生的資料目錄；若本地沒有資料，請依照以下指令下載到固定路徑。`scripts/prepare_lora_dataset.py` 會檢查 `data/raw/cristobalmitchell_pokedex/data/pokemon.csv` 與 `data/raw/cristobalmitchell_pokedex/images/large_images/`，路徑不一致時會直接失敗。

### 1. 下載資料集

推薦使用 GitHub shallow clone，會直接得到 `data/pokemon.csv` 和 `images/large_images/`：

```bash
mkdir -p data/raw
git clone --depth 1 https://github.com/cristobalmitchell/pokedex.git data/raw/cristobalmitchell_pokedex
```

如果 `data/raw/cristobalmitchell_pokedex` 已經存在，不要重複 clone；直接執行下面的結構檢查。

下載完成後先確認結構正確：

```bash
test -f data/raw/cristobalmitchell_pokedex/data/pokemon.csv
test -d data/raw/cristobalmitchell_pokedex/images/large_images
find data/raw/cristobalmitchell_pokedex/images/large_images -maxdepth 1 -type f | head
```

如果使用 Kaggle 下載 The Complete Pokedex Dataset，解壓後也必須整理成相同結構。不要多包一層 `pokedex/`，也不要只把 `images/` 下載到 `data/raw/cristobalmitchell_pokedex/images/` 而缺少 `data/pokemon.csv`。

### 2. 準備 LoRA 資料

```bash
python scripts/prepare_lora_dataset.py \
  --raw-image-dir data/raw/cristobalmitchell_pokedex \
  --resolution 512
```

成功後應產生或更新：

```text
data/processed/lora_images/
data/processed/captions.jsonl
data/processed/annotations.jsonl
data/processed/metadata.json
```

### 3. 訓練 LoRA

使用目前設定：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_sdxl.py \
  --config configs/lora_sdxl.yaml \
  --output-dir outputs/lora/pokecreature_sdxl_lora_sks_anchor_r16_a16_20260623
```

### 4. 切換 app 使用重訓權重

```bash
cp outputs/lora/pokecreature_sdxl_lora_sks_anchor_r16_a16_20260623/pytorch_lora_weights.safetensors \
  weights/pokecreature_sdxl_lora_sks_anchor_r16_a16_20260623/pytorch_lora_weights.safetensors
```

也可以直接修改 `configs/app.yaml` 的 `generation.lora_path` 指向新的 LoRA 輸出目錄。

## 前端部署模式

`configs/app.yaml` 預設：

```yaml
ui:
  deployment_mode: true
  show_debug_panel: false
```

部署模式行為：

- 不顯示本地 LoRA path。
- 不顯示本地 image path。
- 不顯示 IP-Adapter reference path。
- 不顯示 raw LLM JSON。
- 只保留必要操作：輸入條件、生成、進化、退化、調整 seed/steps/guidance。
- 進化與退化預設使用 IP-Adapter。

## 驗證指令

```bash
python -m compileall src scripts app.py
pytest
```

目前本地驗證狀態：

- `python -m compileall src scripts app.py` 通過。
- `pytest` 通過，包含 prompt、schema、lineage、LoRA loader、IP-Adapter loader、UI stage cache 測試。

## 目前 Demo 輸出

最新 LoRA 權重：

```text
weights/pokecreature_sdxl_lora_sks_anchor_r16_a16_20260623/
```

最新 18 屬性 contact sheet：

```text
outputs/demo/type_prompt_sweep_sks_anchor_r16_a16_20260623/20260623_130148_contact_sheet.png
```

目前觀察：

- `sks style` anchor LoRA 明顯提升手繪怪獸風格、線稿與屬性視覺。
- fire、grass、electric、ice、psychic、ghost、steel、fairy 的屬性特徵較穩定。
- water、bug、rock、dragon 仍偶爾出現多視角或 reference-sheet 形式。
- `blank background` 較常學成灰底，還不是完全乾淨白底。

## 限制

- SDXL 與 LoRA 訓練需要 CUDA GPU 才有實用速度。
- LoRA 仍會受到資料集裡官方角色、多視角、reference sheet 先驗影響。
- 數值型六圍 token 對圖像比例的控制仍有限，因為 frozen text encoder 對 `attack150` 這類 token 沒有天然語意。
- IP-Adapter 能提升進化鏈一致性，但 reference scale 太高時會降低進化/退化差異。
- 生成結果需要人工檢查，避免過度接近官方角色。

## 開發紀錄

完整開發過程、關鍵 prompt、工具組合與 Agent 協助排除的技術問題記錄在 `workflow_log.md`
