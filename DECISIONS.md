# 專案決策記錄

給之後在這個 repo 裡工作時參考的脈絡，避免重新討論已經定案的問題。跟研究方法本身相關的
決策（RAE encoder 選型、資料規模等）記在 [`meanflow_rae/README.md`](meanflow_rae/README.md) §10。

## 資料夾結構

- `references/huggingface-pytorch-xla/` 原封不動保留、不修改——當作已驗證的
  PyTorch/XLA + GSPMD TPU 訓練參照；`meanflow_rae/` 需要類似邏輯時複製改寫，不 import。
- `tpu_provisioning/` 獨立放在根目錄：跨子專案共用，不屬於 `meanflow_rae/`；也不用 `infra/`
  包一層——目前算力來源只有 TRC 這一個，多包一層沒意義。
- 根目錄命名 `tpu-image-gen`（不是 `meanflow-rae-tpu`）：涵蓋 `references/` 等非 MeanFlow
  專屬的內容。

## 技術選型

- 訓練框架：PyTorch/XLA（沿用 `references/` 既有風格），不是 JAX/Flax。
- Representation 方法：RAE 取代傳統 VAE；REPA 列為後續消融方向，不在主線。

有變動時同步更新這裡與 `meanflow_rae/README.md` §10。
