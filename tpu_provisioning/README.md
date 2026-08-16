# tpu_provisioning

把 TRC 核准的 TPU 配額轉換成可執行的 `gcloud` 指令，並在 spot 容量被搶佔後自動偵測、
補送申請。

## 檔案

| 檔案 | 用途 |
| --- | --- |
| `trc_quota.py` | 設定檔：目前的 TRC 配額清單、spot 拆分規則、runtime version 對照表、zone 縮寫對照表（`ZONE_ABBREV`，拿來組 slice 名稱）；`PROJECT_ID`/`GCLOUD_ACCOUNT` 從環境變數讀，不寫在檔案裡 |
| `plan_tpu_requests.py` | 讀 `trc_quota.py`，印出（或用 `--run` 執行）`gcloud queued-resources create` 指令，並把 network setup（Private Google Access/Cloud NAT）/check/create/describe/delete 指令寫到 `tpu_commands.md`（各自一個 code block，用 `#` 註解分開每條） |
| `reconcile.py` | 單次比對「應該存在的 slice」vs. 實際狀態，缺的補建、壞的刪除；搭配 cron/systemd 定期執行 |

## 使用前

- 已安裝並登入 `gcloud`（`gcloud auth login`，且該帳號對目標 GCP project 有 TPU 權限）
- 設定環境變數（帳號相關資訊一律不寫進檔案，用 env var 傳入）：

  ```bash
  export TPU_PROJECT_ID=my-gcp-project        # 必填，兩支腳本沒設就會直接報錯退出
  export TPU_GCLOUD_ACCOUNT=bot@my-gcp-project.iam.gserviceaccount.com  # 選填，
      # 沒設就用 gcloud 當下 active 的帳號；設了會在每條指令加上 --account=...，
      # 用來固定成某個帳號（例如共用的 service account）而不依賴機器上的預設登入狀態
  ```

- 確認 `trc_quota.py` 裡的 `QUOTA`（zone/generation/tier/chips）符合目前的 TRC 核准配額——這些是
  區域/配額資訊，不是帳號憑證，所以照舊直接寫在檔案裡
- 如果 `QUOTA` 用到新 zone，記得同步在 `trc_quota.py` 的 `ZONE_ABBREV` 加一筆對照（例如
  `"us-central1-a": "uscent1a"`），沒加會直接報錯退出而不是生出奇怪的名字

## 用法

```bash
# 1. 印出計算好的申請指令（不會執行），同時把 network setup/check/create/describe/delete
#    指令寫到 tpu_commands.md（可用 --output 換路徑）
python plan_tpu_requests.py

# 2. 確認沒問題後送出申請
python plan_tpu_requests.py --run

# 3. 持續監控 spot 是否被搶佔（建議排進 cron，例如每 5 分鐘一次）
python reconcile.py --dry-run   # 先確認邏輯正常，不執行
python reconcile.py             # 實際刪除/補建
```

Slice（Queued Resource）名稱格式是 `trc-{generation}-{每請求chips}-{zone縮寫}-{tier}-{index}`，
例如 `trc-v5e-32-uscent1a-spot-1`（`us-central1-a` 的第 2 個 v5e-32 spot 請求，`index` 從 0
開始編號）。`{每請求chips}` 是拆分後單一請求的晶片數（例如 v5e 拆成 32 一份），不是 `QUOTA`
裡該筆配額的總數。

`--internal-ips` 只加在 spot 的 create 指令上，on-demand 不加。

`tpu_commands.md` 是給人看/複製用的參考文件，Network Setup/Check/Create/Describe/Delete
五段各自一個 ` ```bash ` code block，區塊內每條指令上面用 `# region` 或 `# slice名稱`（或
`# zone`）註解分開，方便掃描與辨識：

- **Network Setup**：Private Google Access（每個 region 一條）+ Cloud NAT（每個 region 兩
  條：先建 router 再建 nat gateway），依 `QUOTA` 用到的 region 自動生成，見上面「網路設定」
- **Check**：`queued-resources list`，每個 zone 一條，只回名稱+狀態，用來快速掃過現況
- **Create**：跟之前一樣
- **Describe**：`queued-resources describe`，每個 slice 一條，回完整細節——某個 slice 卡在
  `WAITING_FOR_RESOURCES` 很久、想知道實際原因時，list 的兩個欄位不夠看，要用這個
- **Delete**：跟之前一樣

指令本身經過 shell-quote（例如 `--format=value(name,state)` 會印成
`'--format=value(name,state)'`），複製後可以直接貼到 Cloud Shell 之類的網頁終端機或本機終端
機跑，不會因為 `(`、`)` 這類字元被 shell 誤判成語法錯誤。不是拿來整份執行的腳本——delete 段
刪的正是 create 段剛建立的同一批 slice，兩段不要連著整段貼上去跑。需要檢查現況、補建或刪除特
定 slice 時，從對應段落複製那兩行（註解 + 指令）出來單獨跑即可。

完整跑一次的實際輸出（14 條指令的具體範例）見 [`example.md`](example.md)。

## 注意事項

### 網路設定

✅ Private Google Access、Cloud NAT 目前 4 個 region 都已確認設定完成，以下是背景說明，供
之後新增 region 時參考。

- **用 `--internal-ips` 之前要先開 Private Google Access**：`build_command()` 幫 spot 加了
  `--internal-ips`，代表 TPU 沒有外部 IP；但建立 TPU 這個動作本身要連 Google 的 API
  （`tpu.googleapis.com` 等），沒外部 IP 的話這條路只能靠 subnet 開 **Private Google Access**
  通，沒開會直接收到 `INVALID_ARGUMENT`（實測 `us-central2` 的 `default` subnet 已確認會擋）。
- **TPU 建起來後要連一般網際網路，需要另外設 Cloud NAT**：跟 Private Google Access 是分開的
  兩件事——那個管「建立 TPU 時連 Google API」，這個管 TPU VM 跑起來之後要 `pip install`、下載
  HuggingFace 模型/資料集這類**一般網際網路**流量，沒外部 IP 的機器預設連不出去，要靠 Cloud
  NAT 才能連。

  兩者都是 **region 層級的一次性設定**，不是每次建 TPU 都要重跑。`plan_tpu_requests.py` 每次
  執行都會依 `QUOTA` 用到的 region，把這兩者的指令自動生成到 `tpu_commands.md` 的
  `## Network Setup` 段（分 `### Private Google Access` / `### Cloud NAT` 兩個子段），不用
  自己手動兜 `--region`。執行前可以先確認是不是已經設過，避免重複執行：

  ```bash
  gcloud compute networks subnets describe default --project=<id> --region=<region> \
    --format="value(privateIpGoogleAccess)"
  gcloud compute routers list --project=<id> --filter="region:<region>"
  ```

- Compute Engine 的 **`INSTANCES` 配額**（per-region，跟 TRC 的 TPU 晶片配額是分開的兩件事）
  可能不夠：實測 `europe-west4` 的 v6e-16 create 因為 `You have reached INSTANCES limit` 而
  `FAILED`。要去 Console「IAM & Admin → Quotas」查對應 region 的 `Instances` 用量並申請調高。

### `plan_tpu_requests.py`

- `--accelerator-type`（如 `v5litepod-32`）、`--runtime-version`（如
  `v2-alpha-tpuv5-lite`）是照過去文件寫的預設值，沒有即時驗證，第一次用之前用
  `gcloud compute tpus accelerator-types list --zone=<zone>` 和
  `gcloud compute tpus versions list --zone=<zone>` 核對。
- `trc_quota.py` 裡 `MAX_CHIPS_PER_SPOT_REQUEST` 的拆分數字是經驗觀察（v5e 大 pod 申請常失敗、
  拆小份就成功），不是 GCP 文件化的 API 限制，之後容量狀況變了可能要重新調整。

### `reconcile.py`

- `HEALTHY_STATES`（判斷 slice 健康與否的狀態字串）目前是 `{"ACTIVE", "PROVISIONING",
  "WAITING_FOR_RESOURCES", "CREATING"}`。實測已確認 `ACTIVE`（`trc-v4-32-uscent2b-spot-0`
  等）、`FAILED`（配額不夠/該 zone 沒容量/內部錯誤都會是這個狀態）、`SUSPENDED`
  （`stateInitiator=SERVICE`，代表 spot 被搶佔，**不會自動恢復**，要刪除重建——`SUSPENDED`
  故意不在 `HEALTHY_STATES` 裡，現有邏輯已經是對的，不用改）。`WAITING_FOR_RESOURCES` 還沒
  實測驗證過，正式排 cron 前建議留意一下。
- 該跑在哪台機器上（獨立常駐控制機 or 某個 TPU VM 自己）尚未定案，見
  [`../meanflow_rae/README.md` §10](../meanflow_rae/README.md#10-開放問題--待決策事項)。
