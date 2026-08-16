# MeanFlow + RAE 圖像生成模型 — TPU (TRC) 訓練藍圖

> 狀態：規劃階段 (blueprint)。本文件描述整體架構、資料/訓練/推論流程與待決策事項，
> 尚未包含實作程式碼。規模 (資料集大小、模型參數量、是否文字條件化) 尚未決定，
> 全文以「先驗證小規模 pipeline、再視結果擴大」為預設策略。

## 0. 專案目標

在 TPU Research Cloud (TRC) 提供的 TPU 資源上，訓練一個結合以下兩個方法的圖像生成模型：

- **MeanFlow**：不預測瞬時速度場，而是預測區間 `[r, t]` 上的「平均速度」，透過
  MeanFlow Identity（JVP-based 一致性條件）訓練，讓模型可以用極少步數（1-step 或
  few-step）取樣，不需要額外的 distillation 階段。
- **RAE (Representation Autoencoder)**：不使用傳統 VAE 的 pixel-reconstruction latent，
  而是直接用一個凍結的預訓練語義編碼器（如 DINOv2 / SigLIP）的特徵當作 latent 空間，
  搭配一個輕量、可訓練的 decoder 把 latent 還原成像素。生成骨幹 (DiT) 直接在這個語義
  latent 空間上做 MeanFlow 訓練。

兩者疊加的核心假設：RAE latent 語義結構更好、更容易學，MeanFlow 讓取樣步數大幅下降，
兩者相加預期可以在有限 TPU 預算（TRC）下用更少的 wall-clock time 得到可用的生成模型。

REPA（在 DiT 中間層對已有 VAE latent 做表徵對齊）作為之後的備案/消融方向記錄在
[§10 開放問題](#10-開放問題--待決策事項)，目前不列入主線。

## 1. 系統流程總覽

```
                         ┌────────────────────────┐
   raw images/text  ───▶ │  資料蒐集 / 前處理 pipeline │
                         └───────────┬────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │  RAE Encoder (frozen)   │  DINOv2 / SigLIP 等預訓練模型
                         │  image ─▶ semantic latent│
                         └───────────┬────────────┘
                                     ▼ (離線快取 latent，節省重複算 encoder 的成本)
                         ┌────────────────────────┐
                         │  DiT + MeanFlow Head     │  訓練目標：平均速度場 u(z_t, r, t)
                         │  (TPU 上訓練的主體)        │
                         └───────────┬────────────┘
                                     ▼ 推論時：1-step / few-step ODE 取樣
                         ┌────────────────────────┐
                         │  RAE Decoder (trainable) │  semantic latent ─▶ pixel image
                         └────────────────────────┘
```

RAE encoder 全程凍結，只有 decoder 與 DiT backbone 需要訓練/微調。

## 2. 資料蒐集 (Data Collection)

| 項目 | 規劃 |
| --- | --- |
| 起步資料集 | ImageNet-1k（或同量級公開資料集）驗證 pipeline 是否收斂 |
| 擴大方向 | 視需要導入 text-to-image 資料（LAION 子集 / DataComp / COYO），目前 TBD |
| 過濾 | 解析度下限、aesthetic score、NSFW/重複影像過濾（依實際資料集決定是否需要） |
| Caption | 若擴充到文字條件生成，需要 caption；沒有 caption 的資料集用現成 captioner 補標 |
| 儲存格式 | WebDataset (tar shards)，避免 TPU host 端隨機存取小檔案造成 I/O 瓶頸 |
| 儲存位置 | GCS bucket（TPU VM 直接掛載/串流讀取） |
| Latent 快取 | RAE encoder 凍結 ⇒ 可離線預先算好所有樣本的 latent，訓練時直接讀快取，省下每個 step 重跑 encoder 的算力，也能用比訓練用的 TPU 更小的機器（CPU/GPU/小 TPU）跑 |
| 工具 | HF datasets / `hf` CLI 下載，webdataset 打包 |

**待決策**：目標資料規模、是否上文字條件、caption 來源。目前先假設 class-conditional
或 unconditional 起步，text-to-image 是後續擴充選項。

## 3. 模型架構 (Model Architecture)

### 3.1 RAE

- **Encoder**：凍結的預訓練視覺模型（候選：DINOv2-B/L、SigLIP2），輸出 patch token 網格
  當作 latent。
- **Decoder**：輕量 ViT 或 CNN decoder，把 encoder 的語義 latent 還原成像素。
  - 選項 A：直接使用社群釋出的 RAE decoder 權重（若存在且授權允許）。
  - 選項 B：自行訓練 decoder（reconstruction loss，可選 adversarial/perceptual loss 輔助）。
  - 兩者皆需驗證 reconstruction 品質是否足夠支撐後續生成任務。

### 3.2 生成骨幹：DiT + MeanFlow

- Backbone 沿用 DiT（Diffusion Transformer）架構，把時間條件 `(r, t)` 一起餵入
  （例如透過 AdaLN-Zero 兩個時間 embedding 相加/concat）。
- 訓練目標改為 MeanFlow：模型輸出 `u(z_t, r, t)`，用 MeanFlow Identity
  （對瞬時速度場做 JVP，構造出平均速度場應滿足的一致性方程）作為 loss，
  不需要額外的 teacher model 或多階段 distillation。
- 模型規模：依 TRC 實際可用的 TPU 資源分 S/B/L/XL 幾個檔位，先在最小檔位上跑通再往上擴。
- 條件化方式：class embedding（起步）／text embedding（擴充，需搭配 text encoder，
  如 T5 或 CLIP text tower）。

## 4. 訓練 (Training)

延續 repo 中既有腳本（[`train_text_to_image_xla.py`](../references/huggingface-pytorch-xla/text_to_image/train_text_to_image_xla.py)、
[`train_cm_ct_unconditional.py`](../references/huggingface-pytorch-xla/train_cm_ct_unconditional.py)）採用的
**PyTorch/XLA + GSPMD** 資料並行模式，沿用同一套 TPU 環境設置（見 [§6](#6-tpu--trc-基礎設施)）。

訓練分期規劃：

1. **Stage 0 — RAE decoder（若需自行訓練）**：在小規模資料上訓練 decoder，驗證重建品質。
2. **Stage 1 — Latent 預處理**：對訓練資料集跑一次 RAE encoder，離線快取 latent 到
   GCS/本地磁碟，訓練時直接讀取，不重複跑 encoder。
3. **Stage 2 — MeanFlow 主訓練**：在快取好的 RAE latent 上訓練 DiT + MeanFlow。
   - Optimizer：AdamW，`bf16` 混合精度（沿用現有腳本設定）。
   - Sharding：GSPMD 沿 batch 維度切分（沿用現有 `PER_HOST_BATCH_SIZE` 模式）。
   - EMA 權重、loss/step time 用 wandb 或 tensorboard 記錄。
4. **Stage 3（選配）— Guidance / CFG 相關處理**：若採用 classifier-free guidance，
   評估是否需要額外的 guidance distillation 步驟（MeanFlow 論文本身的作法待確認是否
   已內建於 identity 中，需要在實作前確認）。

## 5. 推論 (Inference)

- 載入訓練好的 DiT+MeanFlow 權重與 RAE decoder。
- 從純噪聲出發，用 1-step 或 few-step（例如 2~4 步）ODE 取樣得到 latent。
- 過 RAE decoder 還原成像素圖。
- Benchmark 項目：
  - 編譯時間 vs. 生成時間（比照 [`../references/huggingface-pytorch-xla/text_to_image/README.md`](../references/huggingface-pytorch-xla/text_to_image/README.md)
    中「compile time」/「generation time」的量測方式）
  - 不同 step 數（1 / 2 / 4 步）下的生成品質 vs. 速度 trade-off

## 6. TPU / TRC 基礎設施

沿用既有 [`text_to_image/README.md`](../references/huggingface-pytorch-xla/text_to_image/README.md) 中已驗證過的流程：

- 用 Queued Resource API 建立 TPU（v4 / v5e / v6e，依 TRC 配額而定，見下方
  「TRC 配額與 Queued Resource 申請」）
- 安裝 `torch` / `torch_xla` 穩定版（避免釘死特定 nightly snapshot 日期）
- 用 GSPMD 做 batch 維度的資料並行

差異點：

- RAE latent 預先快取階段，不一定需要佔用主力訓練用的大 TPU pod，可以用較小資源
  （甚至 CPU/GPU）先把資料處理完，把大 pod 的時間留給 Stage 2 主訓練。
- 需要額外的 GCS bucket 規劃，存放：原始資料 shards、快取後的 latent shards、
  checkpoint、訓練 log。

### 6.1 TRC 配額與 Queued Resource 申請

TRC 核准後給的是「每個 zone / TPU 世代 / spot 或 on-demand」的晶片數配額，例如：

```
64 spot Cloud TPU v5e chips in zone europe-west4-b
64 spot Cloud TPU v6e chips in zone us-east1-d
64 spot Cloud TPU v5e chips in zone us-central1-a
32 spot Cloud TPU v4 chips in zone us-central2-b
32 on-demand Cloud TPU v4 chips in zone us-central2-b
64 spot Cloud TPU v6e chips in zone europe-west4-a
```

**實務發現**：直接用 Queued Resource API 一次申請整份配額大小的單一 pod
（例如單一 v5e-64 或 v6e-64 spot QR）常常申請不下來。改成把同樣的晶片數拆成
多個較小的 QueuedResource（v5e-32 × 2、v6e-16 × 4）分開申請則可行。這是經驗
觀察，不是文件化的 API 限制，之後 TRC 容量狀況改變的話可能需要重新調整。

這個「拆分大小」目前寫在
[`../tpu_provisioning/trc_quota.py`](../tpu_provisioning/trc_quota.py)
的 `MAX_CHIPS_PER_SPOT_REQUEST`（v5e: 32、v6e: 16、v4: 32，v4 因為配額本來就只有
32 顆，還沒實測過更大的量是否也會失敗）。

**⚠️ 注意**：拆成多個獨立 QueuedResource 之後，得到的是多個各自獨立的 TPU
slice，不是一整塊可以直接用單一 GSPMD mesh 涵蓋的大 pod。現有訓練腳本
（`train_text_to_image_xla.py` 等）的 GSPMD sharding 只在單一 slice 內生效；
要把多個小 slice 當成同一個訓練 job 用，需要額外的 multi-slice/multi-host
協調機制（例如把每個 slice 當獨立 data-parallel replica、跑完再做梯度/權重同步），
這部分還沒有實作，列在 [§10 開放問題](#10-開放問題--待決策事項)。短期內比較
務實的用法是：把每個小 slice 當成一個獨立的訓練/實驗單位（例如拿 v5e-32 跑
Stage 2 訓練，另一個 v5e-32 跑消融實驗），而不是硬湊成一個更大的邏輯 pod。

**申請指令產生**：不手動兜 `gcloud` 指令，改用
[`tpu_provisioning/plan_tpu_requests.py`](../tpu_provisioning/plan_tpu_requests.py)：

```bash
# 1. 設定必填環境變數（帳號相關資訊不寫進檔案，見 tpu_provisioning/README.md「使用前」），
#    並確認 tpu_provisioning/trc_quota.py 的 QUOTA 是最新的 TRC 配額
export TPU_PROJECT_ID=my-gcp-project

# 2. 印出計算好的 gcloud 指令（不會真的執行），同時把 check/create/describe/delete
#    完整指令寫到 tpu_provisioning/tpu_commands.md
python tpu_provisioning/plan_tpu_requests.py

# 3. 確認沒問題後，加 --run 實際送出申請（會先跳確認提示；--yes 可跳過）
python tpu_provisioning/plan_tpu_requests.py --run
```

腳本會依 `MAX_CHIPS_PER_SPOT_REQUEST` 把每一筆配額拆成對應數量的
`gcloud compute tpus queued-resources create` 指令，並印出一份彙總表確認
「拆分後晶片總數」有對上「TRC 核准配額」。細節（`--accelerator-type` 對應、
`--runtime-version` 預設值等）見腳本內註解；`--runtime-version` 這類值會隨時間
變動，執行前建議先用 `gcloud compute tpus versions list --zone=<zone>` 核對。

實測發現 `--internal-ips`（spot 建立時預設會加）還需要 subnet 開通 Private Google
Access、之後要連網際網路還需要 Cloud NAT，否則 create 會直接失敗；目前 4 個 region
都還沒設好，是 Phase 0 卡住的主因，細節見
[`tpu_provisioning/README.md` 注意事項](../tpu_provisioning/README.md#注意事項)。

**Spot 被搶佔後的自動補請求**：TRC 配額大部分是 spot，實務上常態是資源被搶走、
需要重新排。`plan_tpu_requests.py` 只負責「第一次申請」，不會持續監控；持續監控/
補請求的邏輯在
[`tpu_provisioning/reconcile.py`](../tpu_provisioning/reconcile.py)：

- 每次執行都是**單次動作**（比對「應該存在的 slice」vs. gcloud 回報的實際狀態），
  不會自己迴圈等待——要持續運作得靠外部排程（cron / systemd timer）定期呼叫，
  例如每 5 分鐘一次。
- 找不到的 slice → 直接補送 create。
- 狀態不健康（不在 `HEALTHY_STATES` 裡）的 slice → 只刪除，不在同一次執行內
  馬上重建，留給下一次執行去補（避免跟 gcloud 刪除的最終一致性延遲互相競速）。
  代價是異常恢復最多要等兩個排程週期，這是刻意的取捨，不是遺漏。
- `HEALTHY_STATES` 是沒有實際帳號可驗證下的合理猜測，正式排到 cron 前，先手動跑一次
  `gcloud compute tpus queued-resources list --zone=<zone> --format="value(name,state)"`
  跟腳本印出的狀態對一下，字串不對要調整。

```bash
python tpu_provisioning/reconcile.py            # 實際刪除/補建
python tpu_provisioning/reconcile.py --dry-run  # 只印出會做什麼，不執行
```

## 7. 專案目錄規劃

```
meanflow_rae/
├── README.md                 # 本檔案
├── data/                      # 資料下載、過濾、打包成 webdataset 的腳本
├── rae/                       # RAE encoder wrapper、decoder 訓練腳本
├── models/                    # DiT + MeanFlow 模型定義
├── configs/                   # 各規模（S/B/L/XL）的訓練設定檔
├── train_meanflow_xla.py      # 主訓練腳本 (PyTorch/XLA)
├── cache_rae_latents.py       # Stage 1：離線算 latent 並快取
└── infer_meanflow.py          # 推論/取樣腳本
```

（目前僅為規劃結構，尚未建立對應程式碼。）

## 8. 評估 (Evaluation)

- 影像品質：FID / IS，若資源允許加測 CMMD（對小 batch 較穩定的新指標）。
- MeanFlow 特有指標：不同取樣步數（1 / 2 / 4-step）下的 FID 曲線，確認少步數取樣的
  品質是否可接受。
- RAE 相關：decoder 重建品質（PSNR/SSIM/LPIPS）作為生成品質的上界參考。

## 9. Roadmap

依實際動手順序排的 5 個步驟；每一步的核取方塊是進入下一步前建議跑完的驗收項目，不是
嚴格的硬性關卡。

### 1. 創建 TPU 機器

目前先以「`plan_tpu_requests.py` 能產生正確指令、手動貼到 Cloud Shell 執行/檢查/刪除」為
足夠，可以往下一步走。spot 容量本來就會波動，`reconcile.py` 這種自動偵測/補送的機制是之後
真的需要長時間無人值守（例如 Stage 2/3 長時間訓練）時才會用到的優化，不是現在的阻塞項——
先手動盯著、需要時手動重跑指令即可，見下面「已延後」。

- [x] TRC 配額 → `gcloud` 指令產生工具（`plan_tpu_requests.py`），含 spot 拆分邏輯
- [x] 4 個 region（`europe-west4`、`us-east1`、`us-central1`、`us-central2`）開通
      Private Google Access（`--internal-ips` 的前置需求，見
      [`tpu_provisioning/README.md` 注意事項](../tpu_provisioning/README.md#注意事項)）
- [x] 4 個 region 設定 Cloud NAT（TPU VM 之後要連一般網際網路下載模型/資料集才用得到）
- [x] 至少 1 個 TPU slice 成功跑到 `ACTIVE` 狀態，驗證申請流程端到端可用——
      `trc-v4-32-uscent2b-spot-0`、`trc-v4-32-uscent2b-on-demand-0` 已確認 `ACTIVE`
- [x] 確認建立失敗的各種真實原因（zone 無容量、`INSTANCES` 配額瞬間衝高後又釋放、GCP 內部
      錯誤、spot 被搶佔後進入 `SUSPENDED` 且不會自動恢復）都是預期內的容量波動，不是工具的
      bug——`europe-west4` 的 `INSTANCES limit` 錯誤查起來 `usage=0`，推測是 14 條 create
      幾乎同時發出瞬間衝高、失敗後又釋放，不需要額外調配額

**已延後**（`reconcile.py` 程式碼已寫好，但先不驗證/排程——目前手動用 `tpu_commands.md`
複製指令貼到 Cloud Shell 就夠用，等真的需要無人值守監控時再回來做）：

- [ ] `reconcile.py` 用今天實測到的真實混合狀態（`ACTIVE`/`FAILED`/`SUSPENDED`）跑一次
      `--dry-run`，確認判斷邏輯跟手動看到的狀態一致，再拿掉 `--dry-run` 跑一次真的執行
- [ ] `reconcile.py` 排定期執行（cron/systemd）；跑在哪台機器仍待決（見 §10）

### 2. 在 TPU 上用 toy dataset 訓練已有架構（例如 SD1.5），並確保可以正確上傳 HuggingFace

目的是把「TPU/XLA 環境是否正常」跟「RAE+MeanFlow 這個新架構本身是否 work」拆開，先用
已知能訓練起來的架構把整條流程走一遍，之後 Stage 3 出問題時才好判斷是環境問題還是模型
問題。詳細操作步驟見 [`toy_validation/README.md`](toy_validation/README.md)。

- [x] 選一個已有實作、已知能訓練起來的架構（例如 SD1.5 fine-tune），沿用
      [`references/huggingface-pytorch-xla`](../references/huggingface-pytorch-xla) 既有腳本，
      跑通 PyTorch/XLA + GSPMD 訓練流程——`trc-v4-32-uscent2b-on-demand-0` 上實測跑完 50 步
- [x] TPU VM 上設好 HuggingFace 認證（token）、建立目標 repo
- [x] 驗證 checkpoint 格式與 model card，確認能正確 push 到 HuggingFace——`verify_upload.py`
      確認 `kyle0518/sd15-tpu-toy-test` 有完整 `model_index.json`/`README.md`/
      `unet`/`vae`/`text_encoder` 子資料夾，4 個 worker 皆驗證通過

**已延後**：Spot 被搶佔中斷後從最新 checkpoint 自動接續（訓練端 checkpoint/resume，見
§10）。這次 toy 驗證改用 on-demand TPU + 只跑 50 步（幾分鐘內結束），直接避開被搶佔的風
險，且沿用的 `diffusers` 參考腳本本身沒有 `--resume_from_checkpoint` 這類機制，要另外實
作。等 §9 第 3 步訓練時間拉長、真的需要用 spot 跑較久的 job 時再處理，不擋這一步。

### 3. 用 toy dataset 訓練 RAE + MeanFlow

- [ ] 確認 RAE 使用哪個預訓練 encoder（DINOv2 vs SigLIP2 等）
- [ ] 確認 RAE decoder：用釋出權重 or 自行訓練
- [ ] 小規模資料集（如 ImageNet 子集）跑通 Stage 0~2，驗證 MeanFlow loss 收斂
- [ ] 驗證 1-step / few-step 取樣品質是否可用
- [ ] 沿用第 2 步驗證過的 checkpoint/resume 機制，確認在 RAE+MeanFlow 這個新架構上一樣可用

### 4. 創建大型 webdataset

- [ ] **先定案**資料規模與是否文字條件化（class-conditional / unconditional /
      text-to-image，見 §10）——這個決策要在動手蒐集資料前確定，蒐錯規格重做代價很高
- [ ] 依決定的規模蒐集/過濾資料，打包成 WebDataset shards
- [ ] 若走文字條件化，補上 caption pipeline（沒有 caption 的資料用現成 captioner 補標）
- [ ] 資料上傳/串流到 GCS，驗證 TPU VM 端讀取效能不是瓶頸

### 5. 正式訓練 RAE + MeanFlow

- [ ] 依 TRC 實際配額決定模型規模檔位（S/B/L/XL）與對應 TPU pod 大小
- [ ] 若所需規模超過單一 slice 算力，需要 multi-slice/multi-host 訓練協調機制（見 §10）；
      未超過的話可以先跳過這項
- [ ] 視第 3 步小規模結果決定是否嘗試 REPA 作為對照/消融
- [ ] 完整訓練 + 評估（FID/IS/CMMD，不同取樣步數下的品質 vs 速度 trade-off）

## 10. 開放問題 / 待決策事項

| 問題 | 目前狀態 |
| --- | --- |
| 訓練框架 | 已定：PyTorch/XLA（沿用現有 repo 風格） |
| Representation 方法 | 已定：RAE（REPA 列為後續消融方向，不在主線） |
| RAE encoder 選型 | 待定（DINOv2 / SigLIP2 / 其他） |
| RAE decoder 來源 | 待定（沿用釋出權重 / 自行訓練） |
| 資料規模與是否文字條件化 | 待定，先以小規模 class-conditional/unconditional 驗證 pipeline |
| 目標 TPU 規模 | 待定，視 TRC 配額與 Stage 2 驗證結果決定 |
| CFG / guidance 是否需要額外處理 | 待定，需在實作前確認 MeanFlow 原論文設計 |
| Multi-slice 訓練協調 | 待定。TRC spot 配額目前只能拆成多個獨立小 slice 申請下來（見 §6.1），要合併成單一邏輯訓練 job 需要額外的 multi-slice/multi-host 協調機制，尚未設計；短期先當獨立訓練單位使用 |
| `reconcile.py` 的排程位置 | 待定。腳本本身只做單次動作，需要外部 cron/systemd 定期呼叫，但要跑在哪台機器上（獨立的常駐控制機、還是某個 TPU VM 自己）還沒決定 |
| 訓練端的 checkpoint/resume | 待定。slice 被搶佔會直接讓訓練 process 中斷，`reconcile.py` 補回資源後，訓練腳本要能自動找到最新 checkpoint 接著跑，目前 `train_meanflow_xla.py` 還沒設計這部分 |

## 11. 參考文獻

- **MeanFlow**：*Mean Flows for One-step Generative Modeling*（2025）—
  平均速度場 + MeanFlow Identity，訓練可一步生成的模型。
- **RAE (Representation Autoencoder)**：以凍結預訓練語義編碼器作為 latent 空間、
  搭配輕量 decoder 取代傳統 VAE 的做法（2025 前後相關工作）。
- **REPA**：*Representation Alignment for Generation: Training Diffusion Transformers
  Is Easier Than You Think*（Yu et al., ICLR 2025）— 在 DiT 中間層對已有 VAE latent
  做表徵對齊正則化，作為後續對照方向。

> 上列文獻的精確 arXiv 連結尚未在此確認，實作前建議自行查證最新版本與正確引用，
> 避免引用到錯誤或過期的版本。
