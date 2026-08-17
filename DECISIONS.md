# 專案決策記錄

給之後在這個 repo 裡工作時參考的脈絡，避免重新討論已經定案的問題。跟研究方法本身相關的
決策（RAE encoder 選型、資料規模等）記在 [`meanflow_rae/README.md`](meanflow_rae/README.md) §10。

## 資料夾結構

- `references/huggingface-pytorch-xla/` 原封不動保留、不修改——當作已驗證的
  PyTorch/XLA + GSPMD TPU 訓練參照；`meanflow_rae/` 需要類似邏輯時複製改寫，不 import。
- `references/RAEv2/`：RAE 論文官方實作（[nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2)），
  直接複製進來、不用 git submodule（理由跟 `huggingface-pytorch-xla` 一樣：不依賴 upstream
  之後還在不在、有沒有改版）。**PyTorch/GPU 實作，不能直接在 TPU 上跑**，用途是給
  `meanflow_rae/` 設計 RAE encoder/decoder 時參照邏輯、改寫成 XLA 版本。**授權是
  CC BY-NC 4.0**（姓名標示-非商業性），比 repo 其他部分常見的授權更嚴格——非商業研究用途
  沒問題，但不能商用，且需保留出處。
  - `references/RAEv2/paper/RAEv2-arxiv-2605.18324.pdf`：論文本體（*Improved Baselines with
    Representation Autoencoders*，Singh et al.，[arXiv:2605.18324](https://arxiv.org/abs/2605.18324)），
    直接下載進來存檔。**論文本身是 CC BY 4.0**（姓名標示）——注意這跟上面 code repo 的
    CC BY-NC 4.0 是兩個不同授權、涵蓋兩個不同東西（論文文字 vs. 程式碼），CC BY 4.0 允許
    重新散布，下載進公開 repo 沒有授權疑慮，只需保留出處（已在此記錄）。
- `tpu_provisioning/` 獨立放在根目錄：跨子專案共用，不屬於 `meanflow_rae/`；也不用 `infra/`
  包一層——目前算力來源只有 TRC 這一個，多包一層沒意義。
- 根目錄命名 `tpu-image-gen`（不是 `meanflow-rae-tpu`）：涵蓋 `references/` 等非 MeanFlow
  專屬的內容。

## 技術選型

- 訓練框架：PyTorch/XLA（沿用 `references/` 既有風格），不是 JAX/Flax。
- Representation 方法：RAE 取代傳統 VAE；REPA 列為後續消融方向，不在主線。

有變動時同步更新這裡與 `meanflow_rae/README.md` §10。
