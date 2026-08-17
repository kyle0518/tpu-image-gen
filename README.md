# tpu-image-gen

在 TPU Research Cloud (TRC) 資源上進行的圖像生成模型研究。主線專案是
**MeanFlow + RAE**：結合 MeanFlow（少步數/一步生成）與 RAE（用預訓練語義編碼器
取代傳統 VAE 的 latent 空間），在 PyTorch/XLA 上訓練。

## 狀態

🚧 規劃階段。目前有完整的架構藍圖與 TPU 申請工具，訓練/推論程式碼尚未開始實作。
細節見 [`meanflow_rae/README.md`](meanflow_rae/README.md)。

## Roadmap

**第 2 步**已完成（SD1.5 toy fine-tune 在 TPU 上跑通、成功上傳 HuggingFace 並驗證），準備
進入第 3 步，完整 checklist 見
[`meanflow_rae/README.md` §9](meanflow_rae/README.md#9-roadmap)。

1. ✅ 創建 TPU 機器（自動監控延後）
2. ✅ 在 TPU 上用 toy dataset 訓練已有架構（例如 SD1.5），並確保可以正確上傳 HuggingFace
3. 🚧 用 toy dataset 訓練 RAE + MeanFlow
4. ⬜ 創建大型 webdataset
5. ⬜ 正式訓練 RAE + MeanFlow

## 專案結構

```
.
├── meanflow_rae/           # 主線專案：MeanFlow + RAE 圖像生成模型
├── tpu_provisioning/       # TRC 配額申請 + spot 被搶佔後的自動補請求
└── references/             # 外部參考實作（HuggingFace PyTorch/XLA 範例、RAE 論文官方實作）
```

各資料夾的用途與規劃理由見 [DECISIONS.md](DECISIONS.md)。

## 快速導覽

| 我想要... | 去看 |
| --- | --- |
| 了解 MeanFlow + RAE 的架構設計、訓練分期、roadmap | [`meanflow_rae/README.md`](meanflow_rae/README.md) |
| 申請 TRC TPU、監控 spot 被搶佔並自動補請求 | [`tpu_provisioning/`](tpu_provisioning/README.md) |
| 參考 TPU 建立、PyTorch/XLA 安裝、GSPMD sharding 的既有範例 | [`references/huggingface-pytorch-xla/text_to_image/README.md`](references/huggingface-pytorch-xla/text_to_image/README.md) |
| 參考 RAE 論文官方實作（PyTorch/GPU，CC BY-NC 4.0） | [`references/RAEv2/README.md`](references/RAEv2/README.md) |
| 了解專案結構背後的規劃理由 | [DECISIONS.md](DECISIONS.md) |
