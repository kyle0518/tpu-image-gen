# Toy 驗證：SD1.5 on TPU + 上傳 HuggingFace

對應 [`meanflow_rae/README.md` §9 Roadmap 第 2 步](../README.md#9-roadmap)。

## 這一步在驗證什麼

不直接碰 RAE+MeanFlow，先用一個**已知能訓練起來**的架構（Stable Diffusion 1.5）、一個
toy dataset、TPU 上跑一輪完整流程，包含把訓練出來的東西**成功上傳到 HuggingFace**。

目的是把兩種風險分開：

- **環境風險**：TPU 建立、PyTorch/XLA 安裝、GSPMD 訓練跑不跑得起來、HuggingFace 認證/上傳
  流不流得通
- **模型風險**：RAE+MeanFlow 這個新架構本身設計對不對、loss 收不收斂

先把環境風險排除掉，之後 §9 第 3 步（用 toy dataset 訓練 RAE+MeanFlow）如果出問題，才能
直接鎖定是模型本身的問題，不用同時懷疑是不是環境沒裝對。

## 這個資料夾裡的檔案

不用自己去 clone HuggingFace 的 `diffusers` repo 找腳本、也不用自己拼訓練指令——這裡已經
準備好可以直接跑的東西：

| 檔案 | 用途 |
| --- | --- |
| `train_text_to_image_xla.py` | 訓練腳本本體，複製自 [`references/huggingface-pytorch-xla/text_to_image/`](../../references/huggingface-pytorch-xla/text_to_image/)（已驗證過的版本，逐字複製、沒有改動一行邏輯） |
| `utils.py` | 訓練腳本用來存 model card 的輔助模組，訓練腳本會 `from utils import save_model_card` 讀它，兩個檔案要放在同一個目錄下 |
| `requirements.txt` | 這次驗證需要的 Python 套件清單 |
| `verify_upload.py` | 訓練完之後拿來確認 HuggingFace repo 真的有東西，不用自己開網頁檢查 |

跟原本 [`references/.../text_to_image/README.md`](../../references/huggingface-pytorch-xla/text_to_image/README.md)
教的做法不一樣的地方：那份文件教你 `git clone` HuggingFace 的 `diffusers` repo 目前的
`main` branch 來取得腳本和 `diffusers` 套件本身。這裡改成：`diffusers` 套件直接從 PyPI
裝穩定版（已經確認 `>=0.39.0` 有腳本用到的所有函式），訓練腳本則用這個資料夾裡**已經驗證
過內容**的版本，不去依賴 upstream repo 現在長什麼樣子——避免 upstream 之後改了檔案位置或
參數，導致這份指南跟著失效。

## 前置需求

- 至少一台 `ACTIVE` 的 TPU（見 [`tpu_provisioning/`](../../tpu_provisioning/README.md)）。
  本專案目前有兩台可用：
  - `trc-v4-32-uscent2b-on-demand-0`（on-demand，不會被搶佔）
  - `trc-v4-32-uscent2b-spot-0`（spot，可能隨時被搶佔）

  **下面步驟用 on-demand 那台**：第一次裝環境、除錯可能要花點時間，spot 中途被搶走的話
  SSH session 會直接斷掉，重來很煩。這個腳本本身也沒有 checkpoint/resume 機制（見下面
  「已知限制」），被搶佔等於整個進度歸零，用 on-demand 直接避開這個風險。

- HuggingFace 帳號，並在 https://huggingface.co/settings/tokens 生一個 **write 權限**的
  token（要能建立 repo、上傳檔案，read-only token 不夠用）。

- 你執行下面指令的機器（例如 Cloud Shell）上要有這份專案的原始碼，因為第 1 步要把
  `toy_validation/` 這個資料夾整個傳到 TPU VM 上。如果 Cloud Shell 上還沒有這份 repo，先
  clone/pull 一份（`git clone https://github.com/kyle0518/tpu-image-gen.git`）。

- 把 `<你的HF帳號>`、`<你的HF write token>` 換成你自己的值——下面指令裡會出現，這是唯一
  需要你自己填的地方。

## 步驟

以下每一步都是**一條完整、獨立的指令**，跟
[`references/.../text_to_image/README.md`](../../references/huggingface-pytorch-xla/text_to_image/README.md)
用的形式一樣：`gcloud ... ssh ... --command='...'`，從 Cloud Shell 執行一次、SSH 進去跑完
就自動斷開，不用維持一個互動式 SSH session、也不用一直記得自己在哪個目錄。

**先在 Cloud Shell 設定這三個變數**（後面每一步都會用到）：

```bash
export TPU_NAME=trc-v4-32-uscent2b-on-demand-0
export PROJECT_ID=trc-project-504304
export ZONE=us-central2-b
```

### 1. 把這個資料夾傳到 TPU VM 上

```bash
gcloud compute tpus tpu-vm scp --recurse \
  meanflow_rae/toy_validation \
  ${TPU_NAME}:~/toy_validation \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all
```

`--recurse` 把整個資料夾（含 5 個檔案）一次傳過去，`--worker=all` 確保 pod 裡每個 host 都
拿到一份（v4-32 具體有幾個 worker 不確定，全部傳保險）。

### 2. 安裝 PyTorch / PyTorch-XLA

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='
pip3 install torch==2.8.0 torchvision "torch_xla[tpu]==2.8.0"
pip3 install --pre torch_xla[pallas] --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
'
```

如果 `2.8.0` 這個版本號已經不是最新的穩定版（`torch_xla` 更新頻繁），去
[pytorch/xla GitHub](https://github.com/pytorch/xla) 查目前的穩定版號再替換。**不要**釘死
特定 nightly 版本日期，那種版本會被下架，之後裝不到會直接失敗。

驗證裝好了：

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='python3 -c "import torch; import torch_xla; print(\"ok\")"'
```

### 3. 安裝訓練腳本的依賴

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='cd ~/toy_validation && pip3 install -r requirements.txt'
```

（`requirements.txt` 裡已經包含 `diffusers>=0.39.0`，不用另外裝。）

### 4. HuggingFace 認證

```bash
export HF_TOKEN=<你的HF write token>

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command="hf auth login --token ${HF_TOKEN}"
```

`--token` 這個 flag 是為了非互動式登入才加的（跳過一般手動貼 token 的提示）；如果你這台
`hf` 版本不支援 `hf auth login --token`，改用舊指令 `huggingface-cli login --token
${HF_TOKEN}` 效果一樣。

### 5. 執行訓練，並上傳到 HuggingFace

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='
cd ~/toy_validation
export XLA_DISABLE_FUNCTIONALIZATION=0
export PROFILE_DIR=/tmp/
export CACHE_DIR=/tmp/
python3 train_text_to_image_xla.py --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 --dataset_name=lambdalabs/naruto-blip-captions --resolution=512 --center_crop --random_flip --train_batch_size=32 --max_train_steps=50 --learning_rate=1e-06 --mixed_precision=bf16 --output_dir=/tmp/trained-model/ --dataloader_num_workers=8 --loader_prefetch_size=4 --device_prefetch_size=4 --push_to_hub --hub_model_id=<你的HF帳號>/sd15-tpu-toy-test
'
```

- `--max_train_steps=50`：故意設很小，這步只是驗證流程通不通，不是真的要練出一個能用的
  模型
- `--hub_model_id`：記得換成你自己的 HuggingFace 帳號
- 想看每一步的 loss 可以加 `--print_loss`，但會拖慢速度（打斷 XLA 的延遲執行批次優化），
  純驗證流程的話不需要加

`push_to_hub` 的上傳邏輯只有 master worker（`xm.is_master_ordinal()`）會執行，多 worker
情況下不會重複上傳，這個不用自己處理。訓練跑完（50 步應該幾分鐘內結束）腳本會自動把
`/tmp/trained-model/` 整個資料夾上傳上去。這條指令會一直卡在 Cloud Shell 直到訓練結束才
返回，是正常的，不是卡住。

### 6. 驗證上傳結果

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
  --command='cd ~/toy_validation && python3 verify_upload.py <你的HF帳號>/sd15-tpu-toy-test'
```

預期輸出最後一行是：

```
OK: repo has model_index.json, README.md, and unet/vae/text_encoder subfolders.
```

看到 `OK` 就代表這一步完成，可以回報進度、更新 `meanflow_rae/README.md` §9 的核取方塊。
如果印出 `FAIL`，訊息會列出缺了什麼檔案/資料夾，通常代表訓練中途某個環節（例如上傳）沒
有真的成功，回頭看第 5 步的執行輸出找錯誤訊息。

## 已知限制

- **這個腳本沒有 checkpoint/resume 機制**（`train_text_to_image_xla.py` 沒有
  `--resume_from_checkpoint` 這類參數）。對這次 toy 驗證沒差（50 步、幾分鐘跑完，用
  on-demand 也不會被搶），但 §9 第 3 步（toy dataset 訓練 RAE+MeanFlow）如果訓練時間拉長、
  改用 spot，就需要额外實作 checkpoint/resume，不能直接沿用這支腳本的邏輯。
- `train_text_to_image_xla.py`/`utils.py` 是複製自 `references/huggingface-pytorch-xla/`
  的快照，不會跟著 upstream `diffusers` repo 自動更新——這是刻意的（見上面「這個資料夾裡的
  檔案」），確保這份指南長期可重跑，但代表 `diffusers` 套件本身之後若有重大變更（例如某個
  函式簽名改了），這裡不會自動跟上，需要時手動重新驗證。
