# 範例：從 TRC 核准信到申請下去

以這份實際拿到的 TRC 配額示範完整流程（用法本身見 [README.md](README.md)）：

```
64 spot Cloud TPU v5e chips in zone europe-west4-b
64 spot Cloud TPU v6e chips in zone us-east1-d
64 spot Cloud TPU v5e chips in zone us-central1-a
32 spot Cloud TPU v4 chips in zone us-central2-b
32 on-demand Cloud TPU v4 chips in zone us-central2-b
64 spot Cloud TPU v6e chips in zone europe-west4-a
```

## 1. 設定環境變數、確認 `trc_quota.py`

這份配額剛好就是 `trc_quota.py` 裡 `QUOTA` 目前寫的內容，不用改。要做的是設定
`TPU_PROJECT_ID`（沒設腳本會直接報錯退出，不會用預留值跑下去）：

```bash
export TPU_PROJECT_ID=my-trc-project   # 換成你自己的 GCP project id
```

`TPU_GCLOUD_ACCOUNT` 這次示範沒設，所以底下指令不會出現 `--account=...`；需要固定用哪個帳號跑
（例如共用的 service account）才需要另外設它，見 [README.md](README.md#使用前)。

之後配額改了，照同樣的 dict 格式改 `trc_quota.py` 的 `QUOTA` 清單就好，其他兩個腳本不用動。

## 2. 先印出指令，不要直接送出申請

```bash
cd tpu_provisioning
python plan_tpu_requests.py
```

用 `TPU_PROJECT_ID=my-trc-project` 實際跑出來的結果：

```
# Plan summary: zone / generation / tier -> num_requests x chips_per_request (quota)
#   europe-west4-b   v5e  spot       2 x 32  = 64   (quota: 64)
#   us-east1-d       v6e  spot       4 x 16  = 64   (quota: 64)
#   us-central1-a    v5e  spot       2 x 32  = 64   (quota: 64)
#   us-central2-b    v4   spot       1 x 32  = 32   (quota: 32)
#   us-central2-b    v4   on-demand  1 x 32  = 32   (quota: 32)
#   europe-west4-a   v6e  spot       4 x 16  = 64   (quota: 64)

gcloud compute tpus queued-resources create trc-v5e-32-euwest4b-spot-0 --project=my-trc-project --zone=europe-west4-b --node-id=trc-v5e-32-euwest4b-spot-0 --accelerator-type=v5litepod-32 --runtime-version=v2-alpha-tpuv5-lite --spot --internal-ips
gcloud compute tpus queued-resources create trc-v5e-32-euwest4b-spot-1 --project=my-trc-project --zone=europe-west4-b --node-id=trc-v5e-32-euwest4b-spot-1 --accelerator-type=v5litepod-32 --runtime-version=v2-alpha-tpuv5-lite --spot --internal-ips
gcloud compute tpus queued-resources create trc-v6e-16-useast1d-spot-0 --project=my-trc-project --zone=us-east1-d --node-id=trc-v6e-16-useast1d-spot-0 --accelerator-type=v6e-16 --runtime-version=v2-alpha-tpuv6e --spot --internal-ips
gcloud compute tpus queued-resources create trc-v6e-16-useast1d-spot-1 --project=my-trc-project --zone=us-east1-d --node-id=trc-v6e-16-useast1d-spot-1 --accelerator-type=v6e-16 --runtime-version=v2-alpha-tpuv6e --spot --internal-ips
gcloud compute tpus queued-resources create trc-v6e-16-useast1d-spot-2 --project=my-trc-project --zone=us-east1-d --node-id=trc-v6e-16-useast1d-spot-2 --accelerator-type=v6e-16 --runtime-version=v2-alpha-tpuv6e --spot --internal-ips
gcloud compute tpus queued-resources create trc-v6e-16-useast1d-spot-3 --project=my-trc-project --zone=us-east1-d --node-id=trc-v6e-16-useast1d-spot-3 --accelerator-type=v6e-16 --runtime-version=v2-alpha-tpuv6e --spot --internal-ips
gcloud compute tpus queued-resources create trc-v5e-32-uscent1a-spot-0 --project=my-trc-project --zone=us-central1-a --node-id=trc-v5e-32-uscent1a-spot-0 --accelerator-type=v5litepod-32 --runtime-version=v2-alpha-tpuv5-lite --spot --internal-ips
gcloud compute tpus queued-resources create trc-v5e-32-uscent1a-spot-1 --project=my-trc-project --zone=us-central1-a --node-id=trc-v5e-32-uscent1a-spot-1 --accelerator-type=v5litepod-32 --runtime-version=v2-alpha-tpuv5-lite --spot --internal-ips
gcloud compute tpus queued-resources create trc-v4-32-uscent2b-spot-0 --project=my-trc-project --zone=us-central2-b --node-id=trc-v4-32-uscent2b-spot-0 --accelerator-type=v4-32 --runtime-version=tpu-ubuntu2204-base --spot --internal-ips
gcloud compute tpus queued-resources create trc-v4-32-uscent2b-on-demand-0 --project=my-trc-project --zone=us-central2-b --node-id=trc-v4-32-uscent2b-on-demand-0 --accelerator-type=v4-32 --runtime-version=tpu-ubuntu2204-base
gcloud compute tpus queued-resources create trc-v6e-16-euwest4a-spot-0 --project=my-trc-project --zone=europe-west4-a --node-id=trc-v6e-16-euwest4a-spot-0 --accelerator-type=v6e-16 --runtime-version=v2-alpha-tpuv6e --spot --internal-ips
gcloud compute tpus queued-resources create trc-v6e-16-euwest4a-spot-1 --project=my-trc-project --zone=europe-west4-a --node-id=trc-v6e-16-euwest4a-spot-1 --accelerator-type=v6e-16 --runtime-version=v2-alpha-tpuv6e --spot --internal-ips
gcloud compute tpus queued-resources create trc-v6e-16-euwest4a-spot-2 --project=my-trc-project --zone=europe-west4-a --node-id=trc-v6e-16-euwest4a-spot-2 --accelerator-type=v6e-16 --runtime-version=v2-alpha-tpuv6e --spot --internal-ips
gcloud compute tpus queued-resources create trc-v6e-16-euwest4a-spot-3 --project=my-trc-project --zone=europe-west4-a --node-id=trc-v6e-16-euwest4a-spot-3 --accelerator-type=v6e-16 --runtime-version=v2-alpha-tpuv6e --spot --internal-ips

Full check/create/describe/delete command reference written to tpu_provisioning/tpu_commands.md
```

檢查彙總表：每一列「拆分後晶片總數 = 」都要等於後面的「quota:」。這份配額六列全部對得上
（64=2×32、64=4×16、64=2×32、32=1×32、32=1×32、64=4×16），代表沒有漏算或算錯，可以往下走。

最後一行寫的 `tpu_commands.md` 除了上面這 14 條 create 指令，還有三段：每個 zone 一條的
`queued-resources list`（Check，只回名稱+狀態，用來快速掃過現況）、每個 slice 一條的
`queued-resources describe`（Describe，回完整細節，某個 slice 卡住想查原因時用）、以及對應
每個 slice 名稱的 `queued-resources delete`（要清掉某個 slice 時複製那條指令跑，不用自己重
組名稱和參數）。Check/Create/Describe/Delete 四段各自放在一個 ` ```bash ` code block 裡，區
塊內每條指令上面都有 `# slice名稱`（Check 段是 `# zone`）註解分開，掃描起來跟一般 shell 腳
本一樣好認，在 GitHub 之類的網頁上瀏覽時整個區塊也有自己的複製按鈕。這份檔案是複製用的參
考，不是拿來整份貼進終端機執行的腳本，四段之間不要整段貼著跑（Create 接著整段跑 Delete 就
是建了又立刻刪掉）。

## 3. 確認沒問題後送出申請

```bash
python plan_tpu_requests.py --run
```

會先列出「要跑 14 條指令，確定嗎？[y/N]」，輸入 `y` 才會逐條真的執行
`gcloud compute tpus queued-resources create`。任何一條失敗會立刻中止，不會硬跑完剩下的
（`subprocess.run(..., check=True)`），失敗訊息會直接印在畫面上。

## 4. 設定 `reconcile.py` 的定期監控

申請完不是結束——spot 會被搶。先手動跑一次確認邏輯正常：

```bash
python reconcile.py --dry-run
```

輸出格式是每個 slice 一行 `OK` / `MISSING` / `UNHEALTHY`，最後列出會刪除幾個、補建幾個。
確認合理之後，排進 crontab（例如每 5 分鐘檢查一次）：

```cron
*/5 * * * * cd /path/to/tpu-image-gen/tpu_provisioning && /usr/bin/python3 reconcile.py >> /var/log/tpu_reconcile.log 2>&1
```

## 這份示範沒有幫你做的事

這個環境沒裝 `gcloud`、也沒有你的 GCP 憑證，上面第 2 步的輸出是用假的
`TPU_PROJECT_ID=my-trc-project` 跑出來的——邏輯本身（拆分數量、彙總表算法）已經跑過，但指令
送到 GCP 之後是否真的申請成功，只有你自己的帳號跑得出來。其他沒驗證過的細節（accelerator
type、runtime version、`HEALTHY_STATES` 等）見 [README.md 的「注意事項」](README.md#注意事項)。
