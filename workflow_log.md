# Workflow Log

本文件記錄 **Pokemon-style image generator** 的主要開發過程、關鍵 Prompt、使用的工具組合，以及 Agent 協助解決的技術問題。

### ChatGPT-1 - 專題可行性與作業規範確認

- Prompt：
  ```text
  作業規範: 一、 作業概述 本作業作為「深度生成模型」課程的總結，評估對大型語言模型 (LLM) 與擴散模型 (Diffusion Models) 或流匹配 (Flow Matching) 技術的綜合掌握程度。專題需高度依賴 AI Agent 與程式碼生成工具，涵蓋從題目發想、系統架構設計、任務拆解到程式碼實作的全週期。最終需產出具備完整功能性與展示介面 (App) 的生成式 AI 專題。 二、 核心技術要求 專題必須包含以下技術之具體應用（至少涵蓋其一並鼓勵兩者結合）： Large Language Models (LLMs)：涵蓋 Prompt Engineering、RAG 架構、API 呼叫或本機開源模型推論。 Diffusion / Flow Matching Models：影像、音訊或 3D 生成，涵蓋 ControlNet、LoRA 權重掛載、Pipeline 客製化或推論加速。 三、 相關工具與資源說明 為支援 Agentic Workflow 與模型推論，以下為建議或可用之工具與運算資源： Ollama：本機端推論框架。用於在本地環境快速部署與執行量化後的開源 LLM（如 GGUF 格式）。提供相容於 OpenAI 規格的 REST API，適合離線開發驗證與端側運算整合。 NVIDIA NIM (NVIDIA Inference Microservices)：基於容器化的微服務推論平台。內建 TensorRT-LLM 等最佳化推論引擎，支援標準 API 介接。適用於需最大化利用企業級 GPU 記憶體頻寬與運算能力的高效能部署環境。 OpenRouter：模型 API 聚合服務。透過單一端點 (Endpoint) 提供存取不同供應商 (如 Anthropic, OpenAI) 及開源社群模型之介面，方便在開發過程中無縫切換模型與管理 Context Window。 Big Pickle：OpenCode 平台提供的實驗性免費大型語言模型代號（底層為 Zhipu AI GLM-4.6，採 MoE 架構，355B 總參數 / 32B 啟用）。具備 200k Context Window，提供 OpenAI 相容 API (Model ID: opencode/big-pickle)，專門針對程式碼生成任務進行最佳化，適用於驅動編程 Agent。 Agent / CLI 工具：包含 claude cli、codex cli、antigravity cli、open code 等命令列工具。同學可利用此類工具將上述推論資源作為 Backend，透過終端機進行自動化程式碼生成與專案結構建置。 四、 專題執行步驟 (Agent Workflow) 請依循以下由 Agent 輔助的開發流程，並完整記錄互動與生成過程： 階段一：發想與企劃 給予 Agent 初始 Context，要求其生成專題提案，確立專題目標、核心功能與預期整合之技術堆疊。 階段二：架構設計與任務拆解 利用 Agent 將選定題目拆解為具體的實作任務，定義系統架構、前後端介接方式與 API 資料交換格式。 階段三：程式碼生成與實作 使用 CLI 工具或 IDE 的 Agent 實作核心邏輯。開發者需擔任系統規劃者，提供精確 Context（例如特定框架版本或環境依賴），引導 Agent 撰寫訓練、微調或推論腳本，並處理除錯過程。 階段四：介面封裝與總結 指示 Agent 快速生成 Gradio、Streamlit 或前端框架介面，整合模型推論後端為可互動之 App，並利用 Agent 輔助撰寫技術文件。我想要做寶可夢生成器，輸入有：屬性（最多兩個），HP/速度/特攻/特防/物攻/物防，這樣有符合專題規範的需求嗎，可行性高嗎？
  ```
- 工具組合：ChatGPT、課程規範分析、LLM / Diffusion 技術路線評估、專題功能拆解。
- Agent 協助解決之技術問題：確認此題目同時涵蓋 LLM prompt planning 與 diffusion image generation，可符合課程要求；將輸入條件定義為屬性、六圍與外觀描述，並判斷可用 LoRA 學習寶可風格。

### ChatGPT-2 - AGENTS.md 與 plan.md 規劃

- Prompt：
  ```text
  幫我撰寫給agents看的AGENTS.md跟plan.md AGENTS.md是整個專案的大方向的準則，像是需要用git版控，每實作幾個功能就要定時commit，以及使用的framework，選擇哪個模型和用什麼資料集微調 plan.md是紀錄要完成這個專案所需的步驟，因此需要詳細的說明 我要搭配的技術路線：- LLM：gemini-2.5-flash 搭配 groq 的免費模型作為fallback - 生圖模型：SDXL - 微調方法：LoRA - 資料集：Kaggle 上的 Pokémon image datasets 搭配 PokeAPI - 前端介面：Streamlit 額外項目：生成後會顯示目前是幾階進化，如果是基礎型，可以透過進化功能，繼續產生一個計劃後的型態，反之如果我輸入一個高階數值的組合產出二階進化的寶可夢，我可以透過退化功能，產生他退化後的型態
  ```
- 工具組合：ChatGPT、專案規格整理、技術路線設計、檔案結構規劃、Agent workflow 規劃。
- Agent 協助解決之技術問題：把專題拆成可執行任務，定義 LLM fallback、SDXL LoRA、資料來源、前端與 lineage metadata 的責任邊界，避免後續實作時技術路線發散。

### ChatGPT-3 - uv 環境管理要求

- Prompt：
  ```text
  我要使用uv venv來管理環境，requirements.txt也需要建立好，這些加到AGENTS.md
  ```
- 工具組合：ChatGPT、Python 3.11、uv、requirements 規劃。
- Agent 協助解決之技術問題：把環境管理方式固定為 uv，避免 Conda / pip / pyproject 混用，讓後續安裝與驗證流程一致。

### Codex-1 - 根據 plan.md 完成 MVP

- Prompt：
  ```text
  根據 [plan.md](plan.md) 完成專案，gemini跟groq的api key還有HF_token後續等待我填上，選擇有空位的GPU來測試，dataset可以參考https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset/data，微調code搭建好後可以直接開始訓練，並觀察相關指標來確保code的正確性
  ```
- 工具組合：Codex、`rg`、`apply_patch`、uv、Streamlit、Pydantic、PyTorch、Diffusers、pytest、CUDA GPU。
- Agent 協助解決之技術問題：建立完整專案骨架，實作 LLM planner、SDXL pipeline、LoRA loader、資料準備、lineage store、Streamlit UI 與測試，並補上可失敗但不崩潰的 deterministic fallback。

### Codex-2 - Base SDXL 與 LoRA 視覺比較

- Prompt：
  ```text
  先使用原始SDXL的權重並使用相同數據比如同一個屬性跟六圍以及相同外觀敘述來產生一張圖片，接著正式開始微調LoRA，最後合併到原始權重後再推論一次比較視覺效果是否提升
  ```
- 工具組合：SDXL base model、Diffusers LoRA loading/fusion、固定 seed、固定 prompt、CUDA inference、comparison output。
- Agent 協助解決之技術問題：建立可比較的 base vs LoRA 推論流程，確認 LoRA 對線稿、風格與屬性表現的影響，並發現 prompt 太長會被 CLIP 截斷。

### Codex-3 - 更換資料集並重建 annotation

- Prompt：
  ```text
  finetune資料集改成https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset，裡面有根據名字整理好，並且一個寶可夢有多個動作，可以根據名字透過pokeapi去搜尋相關metadata，並且重新整理出一份有寶可夢圖片與對應label包含屬性、六圍、外觀描述的annotation jsonl，目前的微調label只有外觀描述不太正確
  把outputs/ 以及 data/ 就的資料跟測試輸出都移除，換成新的資料集跟測試輸出，這樣會比較清楚
  資料集建好後再重新用全部的資料正式finetune，steps數可以拉長一點，並比較視覺效果
  ```
- 工具組合：Kaggle dataset、PokeAPI、資料清理腳本、JSONL annotation、LoRA training、輸出目錄重建。
- Agent 協助解決之技術問題：將「只有外觀描述」的弱 label 改成包含屬性、六圍與外觀描述的訓練資料，改善 LoRA conditioning 與 inference label 對齊問題。

### Codex-4 - 外觀描述來源檢查

- Prompt：
  ```text
  annotation中的appearance description是怎麼生的，因為我觀察起來會跟原始寶可夢些微對不上，這邊盡量使用官方的敘述，可以先透過pokeapi看看有沒有類似的內容可以接取
  ```
- 工具組合：PokeAPI metadata、CSV/JSON 欄位檢查、annotation builder。
- Agent 協助解決之技術問題：釐清自動摘要會產生不準確外觀描述，改優先使用資料來源中的官方或接近官方描述，降低 label 與圖片不一致的問題。

### Codex-5 - Caption 與 Unicode 清理

- Prompt：
  ```text
  training visual summary會有錯誤的地方，像是Abomasnow的敘述有glowing elemental focus，但是事實上沒有，這邊是怎麼總結的
  另外我想確認lora是使用caption作為文字輸入嗎，這樣推論時也是會將前端輸入內容轉為類似的格式嗎？
  另外還有一個問題是annotation會有一些像是\u00e9 \u2019 之類的unicode，這裡要清理一下
  ```
- 工具組合：caption JSONL、annotation JSONL、Python JSON `ensure_ascii=False`、LoRA training dataset 檢查。
- Agent 協助解決之技術問題：確認 LoRA 訓練文字就是 caption，inference 也必須轉成相同格式；移除錯誤的自動 visual summary，並修正 JSON 輸出避免 Unicode escape 影響可讀性。

### Codex-6 - 改用 cristobalmitchell/pokedex

- Prompt：
  ```text
  [https://github.com/cristobalmitchell/pokedex/tree/main](https://github.com/cristobalmitchell/pokedex/tree/main)
  這個資料集有圖片跟description以及六圍相關資訊，將資料集替換成這個，pokeapi也可以移除，annotation跟caption可以直接用裡面的資訊
  替換完後重新用全部資料lora微調一定數量的steps，將權重合併後用相同格式的輸入推論看看效果如何
  ```
- 工具組合：GitHub dataset、Kaggle mirror、`pokemon.csv`、`large_images`、dataset preparation script、LoRA training。
- Agent 協助解決之技術問題：移除名稱對 PokeAPI metadata 的不穩定匹配，改用同一資料集內的圖片與 CSV 欄位，讓圖片、屬性、六圍與描述來源一致。

### Codex-7 - LoRA rank / alpha 探索

- Prompt：
  ```text
  使用不同的lora rank或是alpha來試試看效果如何
  ```
- 工具組合：LoRA rank/alpha sweep、固定 prompt、固定 seed、固定 LoRA scale、loss 與 contact sheet 比較。
- Agent 協助解決之技術問題：比較不同 LoRA 容量對風格吸附與形體漂移的影響，找出 r16/a16 具有較強風格學習能力但也可能放大資料集偏差。

### Codex-8 - r16/a16 長步數訓練並接前端

- Prompt：
  ```text
  選擇r=16/a=16作為lora的超參數，訓練個4000 steps，如果我覺得視覺效果不錯的話，就接到前端
  ```
- 工具組合：`configs/lora_sdxl.yaml`、SDXL LoRA training、checkpoint、training metrics、Streamlit config。
- Agent 協助解決之技術問題：將選定 LoRA 權重接入 app config，加入 LoRA scale 控制，並確保 pipeline 能依 LoRA path/scale 重新載入，避免 stale fused weights。

### Codex-9 - Electric prompt 與單視角問題

- Prompt：
  ```text
  我有自己做一些測試，像是輸入: { 
    "name": "Galvatrode",
    "types": [
      "electric"
    ],
    "evolution_stage": "stage_2",
    "visual_concept": "A squat, heavily armored amphibian, blending features of a turtle and a frog. Its back is covered by a thick, segmented, dark moss-green shell. Its skin is robust and leathery, adorned with glowing electric blue and vibrant yellow crystalline growths that pulse with absorbed energy. Its limbs are short and sturdy, built for stability.",
    "stat_interpretation": {
      "hp": "70: Possesses a sturdy, robust build, capable of enduring prolonged encounters despite not being exceptionally bulky.",
      "attack": "70: Can deliver a decent physical strike, likely a static-charged headbutt or a powerful slam with its reinforced limbs.",
      "defense": "102: Features an incredibly thick, multi-layered shell that forms a natural Faraday cage, highly resistant to physical impacts.",
      "special_attack": "70: Capable of generating moderate electrical discharges, perhaps through its crystalline growths or a concentrated energy burst.",
      "special_defense": "141: Its shell and dense hide are exceptionally insulating, capable of absorbing and dispersing massive amounts of electrical or energy-based attacks without harm. Specialized glands or crystalline structures passively absorb ambient energy.",
      "speed": "32: Its heavy, armored body and sturdy, short limbs are built for stability and defense, making it very slow and deliberate in movement."
    },
    "core_motifs": [
      "Armored amphibian",
      "Electrical insulation",
      "Energy absorption",
      "Slow resilience",
      "Living battery"
    ],
    "color_palette": {
      "body_and_shell": "Deep moss green and earthy browns",
      "underside": "Lighter, duller greens or beige",
      "electric_elements": "Electric blues, vibrant yellows, and hints of purple for crystalline growths and bioluminescent patterns",
      "eyes": "Bright amber or piercing yellow"
    },
    "sdxl_prompt": "A sturdy, squat creature resembling a hybrid of a turtle and a frog. Its back is covered by a thick, segmented, dark moss-green carapace with earthy brown undertones. The skin on its exposed limbs and face is thick, leathery, and dull green, with subtle, warty textures. Along its shell and thick hide, there are glowing electric blue and vibrant yellow crystalline growths and bioluminescent patterns that pulse with static energy. Its eyes are a bright, piercing amber. Its short, powerful limbs are planted firmly on the ground, suggesting immense stability and slowness. Environment: damp, rocky terrain with faint electrical discharge in the air. Pokemon style, creature design, detailed scales, intricate patterns, soft glowing light, volumetric lighting, digital art, high resolution.",
    "negative_prompt": "ugly, deformed, disfigured, poor anatomy, bad composition, blurry, low resolution, poorly drawn, out of frame, extra limbs, missing limbs, human, human-like, text, watermark, signature, cartoon, anime, low-poly, 3d render, plastic, toy, doll",
    "pokedex_entry": "Known for its incredible resilience against energy attacks, Galvatrode's dense shell and specialized skin act as a living insulator. It moves slowly but deliberately, preferring to stand its ground and absorb electrical currents from its environment. The energy it collects is stored within its crystalline growths, occasionally discharging in harmless, bright flashes.",
    "evolution_hint": "Evolves from its previous form after enduring multiple powerful Electric-type attacks in battle, strengthening its natural insulation and energy absorption.",
    "devolution_hint": "To reach its final form, Galvatrode must be exposed to a unique conductive mineral found deep within volcanic caves, allowing it to fully channel and unleash its vast stored electrical energy."
  }
  但是產生的圖並沒有閃電的要素
  而且有時候進化或退化後會出現多視角的角色圖，我只想要有一個視角的視覺圖就好
  ```
- 工具組合：prompt builder、negative prompt merge、SDXL token budget 檢查、LoRA inference、Streamlit/CLI generation。
- Agent 協助解決之技術問題：發現 electric 視覺詞被 prompt 截斷或權重不足，調整 prompt 優先級，加入 visible lightning bolts / electric arcs 等元素，並強化單視角構圖限制。

### Codex-10 - Inference 格式對齊 captions.jsonl

- Prompt：
  ```text
  調整inference輸入的格式以match [captions.jsonl](data/processed/captions.jsonl)，因為lora是用這邊的文字訓練的，調整完嘗試看看每個屬性以及用不同的外觀描述，觀察產生出來的圖有沒有按照prompt生成
  ```
- 工具組合：`src/generation/prompt_builder.py`、caption-style prompt、type sweep、contact sheet。
- Agent 協助解決之技術問題：修正訓練 caption 與推論 prompt 分布不一致的問題，讓 LoRA 在 inference 時看到與訓練更接近的文字格式。

### Codex-11 - raw large_images 與簡化 label

- Prompt：
  ```text
  [processed](data/processed/) 裡面所擷取的image，像是 @data/processed/lora_images/00004.png 背景會有點破圖，直接擷取 @data/raw/cristobalmitchell_pokedex/images/large_images 底下的圖就好，@data/processed/captions.jsonl 裡面的值也要修改，我text只要包含屬性+六圍+外觀描述就好，值都可以直接從 @data/raw/cristobalmitchell_pokedex/data/pokemon.csv 裡面複製過去就好，lora訓練也是用這個當作label，inference時也是從前端接取對應的值來生圖，要保持統一
  
  以上修改完後重新lora微調一版，並用每個屬性推論一遍觀察效果有沒有變好
  ```
- 工具組合：`scripts/prepare_lora_dataset.py`、SHA256 檢查、`pokemon.csv`、SDXL tokenizer 長度檢查、LoRA training、type sweep。
- Agent 協助解決之技術問題：移除重新貼白底造成的背景破圖，讓 processed image 直接等於 raw image；同時建立訓練與前端推論共用的 compact label contract。

### Codex-12 - 最小 negative prompt 與 r16/a16 重訓

- Prompt：
  ```text
  目前生成的圖片跟寶可夢的風格差異非常大，是不是negative prompt太強烈，我只要留：不要多視角或多格子
  另外我要用r=16/a=16再微調一次觀察效果如何
  繼續分析還有哪邊會造成風格沒有學好或是推論的問題
  ```
- 工具組合：prompt template、LoRA r16/a16 training、loss metrics、type sweep、visual inspection。
- Agent 協助解決之技術問題：將 negative prompt 縮到 layout guards，分析風格問題不只來自 negative prompt，也來自資料集 reference-sheet 先驗、label 太稀疏、未訓練 text encoder 與 CLIP token 語意不足。

### Codex-13 - style anchor 與 IP-Adapter

- Prompt：
  ```text
  接下來要改進的部分：
  - 加入style token像是pokemon把寶可夢風格綁到lora，不過要重新想一個詞不然可能原本就有這個語意了，以及移除negative prompt在lora訓練時在label中設計一些anchor token學習到像是單一生物、寶可夢風格、全身呈現、空白背景、乾淨構圖等概念，並在推論中使用這些anchor token觀察有沒有學好
  - 先訓練UNet lora就好，不過training steps可以拉長一點，等待收斂再來比較才知道有沒有學好
  - 我想加入可啟用/關閉的IP-adapter用預訓練的就好，用在進化/退化的功能上，為了保持視覺特徵的一致性
  ```
- 工具組合：custom style token、composition anchors、UNet LoRA、`h94/IP-Adapter`、Diffusers `load_ip_adapter`、Streamlit controls、pytest。
- Agent 協助解決之技術問題：設計 anchor token 讓 LoRA 學風格與構圖概念；修正 IP-Adapter 與 attention slicing 的相容性問題，讓進化/退化能使用 reference image 保持一致性。

### Codex-14 - 移除 negative prompt 並測 IP-Adapter

- Prompt：
  ```text
  把negative prompt移除，我要先看一下每個屬性的視覺話效果，也選一張圖片來進化跟退化，我要看一下一致性是否有效
  ```
- 工具組合：no-negative inference、type sweep、psychic reference image、IP-Adapter lineage test、contact sheet。
- Agent 協助解決之技術問題：確認移除 negative prompt 會讓多視角與 grid artifact 回來；IP-Adapter 可保留顏色、輪廓與 motif，但 scale 太高會降低進化/退化差異。

### Codex-15 - sks style anchor 與 type/stat conditioning 測試

- Prompt：
  ```text
  anchor token的設計：
  - 寶可夢風格的token可以用像是sks style這種本身沒有意義的詞去學
  - 其他的像是單一生物跟空白背景，用逗號分隔並且可以使用single image跟blank background這種具有意義的詞匯
  
  目前六圍跟屬性是不是有用對應的prompt來去讓圖片生成類似刻板印象的形象，我是希望可以讓lora學到每個屬性或是不同六圍的比例分別會對應到什麼樣的圖像分布，這部分是可行的嗎，幫我測試看看效果如何
  
  前面有觀察到移除negative prompt會造成圖片出現像是多視角等我不想出現的情況，之後推論時可以加上最小的negative prompt來優化最後生成結果
  ```
- 工具組合：caption builder、SDXL tokenizer validation、same-seed type probe、same-seed stat probe、LoRA inference。
- Agent 協助解決之技術問題：確認 type token 對顏色、元素效果與局部形體有影響；stat numeric token 效果較弱，原因是 frozen text encoder 對 `attack150` 這類 token 沒有天然語意，且資料量與分布不足以解開 stats、species、type 的糾纏。

### Codex-16 - 新 anchor LoRA 訓練與 18 屬性視覺化

- Prompt：
  ```text
  用新的anchor重新訓練一版lora並視覺化每個屬性的結果
  ```
- 工具組合：SDXL LoRA training、CUDA GPU 0、r16/a16、3000 optimizer steps、training metrics、18-type contact sheet。
- Agent 協助解決之技術問題：完成 `sks style` anchor 版 LoRA，觀察到整體風格吸附明顯提升，但 water/bug/rock/dragon 仍受多主體或 reference-sheet 資料先驗影響。

### Codex-17 - 部署版前端與進化鏈 stage cache

- Prompt：
  ```text
  - 前端要記錄進化狀態，如果有生成過的要存暫存，比如說base/one-stage/two-stage，我第一次生成出來的是one-stage要記錄以生成並把圖片存下來，這樣我進化後按退化可以回去原來的圖，退化也是用原本的圖去當IP
  
  - 目前前端頁面是偏debug模式可以看到很多資訊，可以註解起來或是設計開關，我要轉換成deployment模式，並將前端文字轉成繁體中文，介面裡像是lora path這些會透露本地路徑的資訊都要移除
  ```
- 工具組合：Streamlit session state、`src/ui/state.py`、stage cache、IP-Adapter reference image、Traditional Chinese UI、pytest、HTTP check。
- Agent 協助解決之技術問題：解決退化時重新生成導致無法回到原圖的問題；加入 stage cache，命中已生成階段就直接切換；同時移除前端 LoRA path、image path、raw JSON 等部署時不該曝光的資訊。