# Toy 驗證：SD1.5 on TPU + 上傳 HuggingFace

對應 [`meanflow_rae/README.md` §9 Roadmap 第 2 步](../README.md#9-roadmap)。

## 目的

在動 RAE+MeanFlow 這個新架構之前，先用一個已知能訓練起來的架構（Stable Diffusion 1.5）
搭配 toy dataset，在 TPU 上完整跑一次「訓練 → 上傳 HuggingFace」的流程。

把兩種風險分開驗證：

- **環境風險**：TPU、PyTorch/XLA、GSPMD 訓練、HuggingFace 認證/上傳是否正常運作
- **模型風險**：RAE+MeanFlow 這個新架構本身設計是否正確、loss 是否收斂

先排除環境風險，之後第 3 步（RAE+MeanFlow）出問題時就能直接鎖定是模型本身的問題。

## 這個資料夾裡的檔案

| 檔案 | 用途 |
| --- | --- |
| `train_text_to_image_xla.py` | 訓練腳本本體，複製自 [`references/huggingface-pytorch-xla/text_to_image/`](../../references/huggingface-pytorch-xla/text_to_image/) |
| `utils.py` | 訓練腳本用來存 model card 的輔助模組，需與訓練腳本放在同一個目錄下 |
| `requirements.txt` | 這次驗證需要的 Python 套件清單 |
| `verify_upload.py` | 檢查 HuggingFace repo 上傳結果是否完整 |

跟 [`references/.../text_to_image/README.md`](../../references/huggingface-pytorch-xla/text_to_image/README.md)
的差異：那份文件是 `git clone` HuggingFace `diffusers` repo 的 `main` branch 來取得腳本和
套件本身。這裡改成 `diffusers` 從 PyPI 裝穩定版，訓練腳本則固定用這個資料夾裡的版本，不
依賴 upstream repo 目前的狀態，避免上游檔案位置或參數變動導致這份指南失效。

## 前置需求

- 至少一台 `ACTIVE` 的 TPU（見 [`tpu_provisioning/`](../../tpu_provisioning/README.md)）。
  下面步驟建議用 **on-demand** slice：這個腳本沒有 checkpoint/resume 機制（見「已知限制」），
  用 spot 中途被搶佔會導致進度歸零，第一次跑建議先用 on-demand 排除這個風險。
- HuggingFace 帳號，並在 https://huggingface.co/settings/tokens 生一個 **write 權限**的
  token（需要能建立 repo、上傳檔案）。
- （選用）Weights & Biases 帳號，用來即時追蹤訓練 loss 曲線，見第 5 步。
- 把下面指令裡的 `<你的GCP-project-id>`、`<你的HF帳號>`、`<你的HF write token>` 換成自己的值。

## 步驟

每一步都是一條完整、獨立的指令，採用
[`references/.../text_to_image/README.md`](../../references/huggingface-pytorch-xla/text_to_image/README.md)
的形式：`gcloud ... ssh ... --command='...'`，從 Cloud Shell 執行一次、SSH 進去跑完就自動
斷開，不需要維持互動式 session。

先在 Cloud Shell 設定以下變數（每次開新 session 都要重新 export，之後每一步都會用到）：

```bash
export TPU_NAME=trc-v4-32-uscent2b-on-demand-0
export PROJECT_ID=<你的GCP-project-id>
export ZONE=us-central2-b
```

### 1. 在 TPU VM 上 clone 這個 repo

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='git clone https://github.com/kyle0518/tpu-image-gen.git ~/tpu-image-gen'
```

`--worker=all` 確保 pod 裡每個 host 都 clone 一份。這個 repo 是 public，TPU VM 不需要另外
設定 GitHub 認證。

### 2. 安裝 PyTorch / PyTorch-XLA

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='
pip3 install torch==2.8.0 torchvision "torch_xla[tpu]==2.8.0"
pip3 install --pre torch_xla[pallas] --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
'
```

如果 `2.8.0` 已經不是目前的穩定版（`torch_xla` 更新頻繁），去
[pytorch/xla GitHub](https://github.com/pytorch/xla) 查目前穩定版號再替換。不要釘死特定
nightly 版本日期，那類版本會被下架。

驗證安裝：

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='python3 -c "import torch; import torch_xla; print(\"ok\")"'
```

### 3. 安裝訓練腳本的依賴

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='cd ~/tpu-image-gen/meanflow_rae/toy_validation && pip3 install -r requirements.txt'
```

`requirements.txt` 故意不含 `peft`：訓練腳本沒用到 PEFT/LoRA，裝了不符合 `diffusers` 版本
要求的 `peft` 反而會讓 `import diffusers` 失敗（見「已知限制」）。如果環境裡已經裝過舊版
`peft`，先解除安裝：

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='pip3 uninstall -y peft'
```

### 4. HuggingFace 認證

```bash
export HF_TOKEN=<你的HF write token>

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command="python3 -c \"from huggingface_hub import login; login(token='${HF_TOKEN}')\""
```

用 `huggingface_hub.login()` 而不是 `hf auth login` CLI：CLI 是 pip 裝的 console script，
裝在 `~/.local/bin`，非互動式 SSH session 預設不會把這個路徑加進 `PATH`。直接呼叫
`huggingface_hub` 套件的 `login()` 函式（`hf auth login` 底層也是包這個函式）不依賴 `PATH`。

### 5. 設定 Weights & Biases（訓練過程紀錄）

在 https://wandb.ai/authorize 取得 API key，然後：

```bash
export WANDB_API_KEY=<你的W&B API key>

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command="WANDB_API_KEY=${WANDB_API_KEY} python3 -c \"import wandb; wandb.login()\""
```

`wandb.login()` 會讀取 `WANDB_API_KEY` 環境變數完成認證，效果等同互動式的 `wandb login`，
但不需要終端機輸入。認證只需要做一次，之後 `wandb.init()`（第 6 步的訓練腳本裡）會自動
沿用同一組認證。

這一步是選用的：不設定的話，第 6 步不要加 `--report_to=wandb`，訓練腳本就完全不會碰
`wandb`，行為跟原本一樣。但建議在這個 toy 階段就測過一次，把「W&B 認證能不能過」這種環境
風險也一併驗證掉，而不是留到第 3 步正式訓練時才第一次踩到。

### 6. 執行訓練，並上傳到 HuggingFace

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='
cd ~/tpu-image-gen/meanflow_rae/toy_validation
export XLA_DISABLE_FUNCTIONALIZATION=0
export PROFILE_DIR=/tmp/
export CACHE_DIR=/tmp/
python3 train_text_to_image_xla.py --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 --dataset_name=lambdalabs/naruto-blip-captions --resolution=512 --center_crop --random_flip --train_batch_size=32 --max_train_steps=50 --learning_rate=1e-06 --mixed_precision=bf16 --output_dir=/tmp/trained-model/ --dataloader_num_workers=8 --loader_prefetch_size=4 --device_prefetch_size=4 --push_to_hub --hub_model_id=<你的HF帳號>/sd15-tpu-toy-test --print_loss --report_to=wandb --wandb_project=tpu-image-gen --wandb_run_name=toy-sd15
'
```

- `--max_train_steps=50`：故意設很小，這步只驗證流程通不通，不是要練出可用的模型。
- `--hub_model_id`：換成自己的 HuggingFace 帳號。
- `--print_loss`：每一步印出 `Step: X, Loss: Y` 到 Cloud Shell，讓指令執行期間看得到進度。
- `--report_to=wandb`：把 loss/learning rate 每 `--logging_steps`（預設 10）步記錄一次到
  W&B，可以在 https://wandb.ai 上即時看曲線，SSH session 斷掉之後歷史紀錄也還在。沒跑過第
  5 步（W&B 認證）的話拿掉這個 flag 即可，訓練照樣正常執行。
- 兩者都是透過 `xm.add_step_closure()` 延遲讀取數值，避免打斷 XLA 的批次優化；`print_loss`
  是逐步印、`--logging_steps` 控制 W&B 的記錄頻率，訓練規模變大時只需要調高
  `--logging_steps`，不需要改程式碼。

`push_to_hub` 只有 master worker（`xm.is_master_ordinal()`）會執行，多 worker 不會重複
上傳。這條指令會一直卡在 Cloud Shell 直到訓練結束才返回，是正常行為。

### 7. 驗證上傳結果

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='cd ~/tpu-image-gen/meanflow_rae/toy_validation && python3 verify_upload.py <你的HF帳號>/sd15-tpu-toy-test'
```

預期輸出最後一行：

```
OK: repo has model_index.json, README.md, and unet/vae/text_encoder subfolders.
```

印出 `FAIL` 代表缺了什麼檔案/資料夾，回頭看第 6 步的執行輸出找錯誤訊息。

## 已知限制

- **沒有 checkpoint/resume 機制**：`train_text_to_image_xla.py` 沒有
  `--resume_from_checkpoint` 這類參數。這次 toy 驗證沒差（50 步、用 on-demand），但第 3 步
  如果訓練時間拉長、改用 spot，需要另外實作 checkpoint/resume。
- `train_text_to_image_xla.py`/`utils.py` 是複製自 `references/huggingface-pytorch-xla/`
  的快照，不會跟著 upstream `diffusers` repo 自動更新。`diffusers` 套件本身之後若有重大變更，
  這裡不會自動跟上，需要時手動重新驗證。
- **`requirements.txt` 不裝 `peft`**：`diffusers` 匯入時會檢查已安裝的 `peft` 版本（目前
  門檻 `>=0.17.0`），版本不夠新會讓 `import diffusers` 直接失敗。訓練腳本沒有用 PEFT/LoRA，
  故不裝；如果之後改用需要 LoRA 的腳本，要另外裝 `peft>=0.17.0`。
- **`--report_to=wandb` 只記錄 loss/learning rate 曲線，不含生成樣本圖**：diffusion 模型的
  loss 數字不一定能直接反映生成品質，定期跑推論存生成圖比對是後續步驟才會需要的功能，目前
  沒有實作在這支腳本裡。RAE+MeanFlow 的訓練腳本（第 3 步）設計時要一併考慮。
