# YOLOv9 聯邦式學習框架

一個基於 YOLOv9 的聯邦式學習系統，使用 Slurm 叢集環境進行分散式訓練和聚合。

## 📋 快速導覽

### 一、說明
- [系統概述](#-系統概述)
- [環境需求](#-環境需求)
- [安裝與設定](#-安裝與設定)
- [目錄結構](#-目錄結構)

### 二、執行
- [快速開始 (自動模式)](#-快速開始-自動模式)
- [實驗續跑 (Replay)](#-實驗續跑-replay)
- [Standalone 模式 (無 Slurm)](#-standalone-模式-無-slurm)
- [單元測試](#-單元測試)

### 三、驗證與補充
- [模型驗證說明](#-模型驗證說明)
- [監控與偵錯指南](#-監控與偵錯指南)
- [支援的聚合演算法](#-支援的聚合演算法)

---

## 🎯 系統概述

本框架旨在使用國網中心的 HPC 叢集環境 (TWCC / N5)，實現一個完整的聯邦式學習 (Federated Learning) 流程。

本專案預設使用初始模型權重 (`yolov9-c.pt`) 做pretrai，分發給多個客戶端 (Client)，客戶端在各自的資料子集上進行訓練後，將更新後的權重傳回伺服器進行聚合 (Federated Averaging)，產生新一輪的全域模型。此過程會重複多個輪次，以期在保護資料隱私的前提下，訓練出一個高效能的全域模型。

* 聯邦學習流程示意
```
Round 1: yolov9-c.pt → [Client1, Client2, Client3, Client4] → w_s_r1.pt
Round 2: w_s_r1.pt  → [Client1, Client2, Client3, Client4] → w_s_r2.pt
Round 3: w_s_r2.pt  → [Client1, Client2, Client3, Client4] → w_s_r3.pt
...
```
![FL Workflow](pics/fl_hpc_overview.gif)

---

## 🛠️ 環境需求
- **執行環境** : NCHC [TWCC](https://www.nchc.org.tw/Page?itemid=6&mid=10)
  - **作業系統**: Linux
  - **作業調度器**: Slurm Workload Manager
  - **容器引擎**: Singularity
  - **Python**: ≥ 3.8
  - GPU: NVIDIA GPU (支援 CUDA)
  - **Pytorch**: PyTorch (≥ 2.1.0)
  - 實驗追蹤 : Wandb 

---

## 📦 安裝與設定

### 1. 取得專案與子模組
```bash
# Clone 主專案
git clone <repository-url>
cd fl_yolo_slurm

# 下載 yolov9 模組
# (也可直接  git clone https://github.com/WongKinYiu/yolov9.git  )
git submodule update --init --recursive
```

### 2. 準備必要檔案
確保以下檔案已放置於專案根目錄：
- **Singularity 映像檔**: `yolo9t2_ngc2306_20241226.sif` ([twcc-cos載點](https://cos.twcc.ai/wauehpcproject/yolo9t2_ngc2306_20241226.sif))
- **初始權重**: `yolov9-c.pt` ([official載點](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt))

### 3. 準備資料集
將您的資料集放置在 `datasets/` 目錄下，並在 `data/` 中建立對應的 `.yaml` 設定檔。
- **[📖 資料集準備指南](./readme_datasets.md)**

### 4. 目錄結構
```
.
├── README.md               # 說明文件
├── readme_sop.md           # 📖 手動執行 SOP 指南
├── readme_val.md           # 📊 模型驗證指南
├── readme_debug.md         # 🔍 偵錯指南
├── yolov9/                 # YOLOv9 原始碼 (Git Submodule)
├── src/                    # 主要目錄
│   ├── orchestrate.sh      # 主程式
│   └── ...                 # 其他輔助腳本
├── data/                   # 資料集 YAML 設定檔
├── datasets/               # 放置原始資料集
├── federated_data/         # 存放分割後的客戶端資料
├── experiments/            # 所有實驗結果的輸出根目錄
│   └── {EXP_ID}/
├── yolo9t2_ngc2306_20241226.sif    # Singularity 容器
└── yolov9-c.pt             # 初始預訓練權重
```



---

## 🚀 快速開始

### 全自動模式 (Slurm 叢集)
```bash
# 方式 1: 使用 sbatch (所有工作都在工作節點執行)
sbatch src/run.sb 

# 方式 2: 使用 orchestrate.sh (少數工作在登入節點執行)
./src/orchestrate.sh kitti 4 2
```
> **提示**: 若要包含最終的模型驗證，請加上 `--val` 旗標。

執行畫面會自動偵測是否要分割資料集，然後發起 n+1 個 Slurm 程序：
- n 個 client train (parallel)
- 1 個 server aggregate (waiting for client complete)

![slurm](pics/sim10k_c4_r5_slurm.png)

---

## � 實驗續跑 (Replay)

如果聯邦學習實驗中途失敗，可以使用 `replay.sh` 從中斷點繼續執行：

```bash
# 指定實驗目錄，自動檢測已完成的輪次並繼續
./src/replay.sh experiments/18_kitti_4C_6R_202510010849
```

**功能特點**：
- ✅ 自動檢測已完成的輪數
- ✅ 從失敗點繼續執行
- ✅ 完整的日誌記錄
- ✅ 避免重複執行已完成的輪次

---

## 💻 Standalone 模式 (無 Slurm)

適合在本地環境或沒有 Slurm 的伺服器上進行測試：

```bash
# 所有參數從 env.sh 讀取
./src/standalone_orchestrate.sh

# Dry-run 模式：只顯示命令，不實際執行
./src/standalone_orchestrate.sh --dry-run
```

**特點**：
- ✅ 不需要 Slurm 環境
- ✅ 順序執行所有客戶端訓練
- ✅ 適合小規模測試和除錯
- ✅ 支援 dry-run 預覽模式

---

## 🧪 單元測試

快速測試聚合演算法和訓練流程：

```bash
# 編輯 src/run_unit_test.sh 設定 EXP_ID 和演算法
# 然後執行
./src/run_unit_test.sh
```

**測試流程**：
1. 使用既有實驗的 Round 1 客戶端輸出進行聚合測試
2. 使用聚合後的權重進行 Round 2 客戶端訓練測試
3. 驗證 loss 值是否正常、NaN/Inf 處理是否正確

---

### 📖 手動模式的進階說明 (SOP)
**注意**: Manual SOP 模式已在 v3 中移除，改為專注於自動化流程。
如需詳細控制，請使用 `standalone_orchestrate.sh` 或直接參考腳本內容。

---

## 📊 模型驗證說明

本系統提供對聯邦學習模型效能的完整分析。詳細的啟用與操作方式請參考：
- **[📊 模型驗證說明文件](./readme_val.md)**

![validation](pics/kitti_c4_r3_val.png)

---

## 🔍 監控與偵錯指南

提供 Slurm 監控、日誌檢查和常見問題的解決方案。詳細內容請參考：
- **[🔍 監控與偵錯指南](./readme_debug.md)**

---

## 🧮 支援的聚合演算法

本框架支援多種先進的聯邦學習聚合演算法，可在 `src/env.sh` 中設定 `SERVER_ALG` 變數：

| 演算法 | 說明 | 適用場景 | 超參數 |
|--------|------|---------|--------|
| **fedavg** | 標準聯邦平均 | 通用，IID 數據 | - |
| **fedprox** | FedProx (近端項正則化) | Non-IID 數據 | `SERVER_FEDPROX_MU` |
| **fedavgm** | FedAvgM (Server 端動量) | 加速收斂 | `SERVER_FEDAVGM_LR`, `SERVER_FEDAVGM_MOMENTUM` |
| **fedopt** | FedOpt (Server 端 Adam) | 穩定訓練 | `SERVER_FEDOPT_LR`, `SERVER_FEDOPT_BETA1`, `SERVER_FEDOPT_BETA2` |
| **fedyoga** | **FedYOGA (自適應權重)** | **Non-IID 數據，不均衡** | `SERVER_FEDYOGA_PCA_DIM`, `SERVER_FEDYOGA_CLIP_THRESHOLD` 等 |
| **fednova** | FedNova (標準化聚合) | 異質訓練步數 | `SERVER_FEDNOVA_MU`, `SERVER_FEDNOVA_LR` |

### FedYOGA 特色功能

**FedYOGA** 是本框架的進階聚合演算法，特別針對 Non-IID 和數據不均衡場景優化：

- ✅ **PCA 降維**: 減少權重差異的維度，提升聚合效率
- ✅ **自適應權重**: 根據客戶端的 loss drop 和 gradient variance 動態調整權重
- ✅ **數值穩定性**: 
  - 自動檢測並修復 BatchNorm 統計量中的 NaN/Inf
  - 跳過損壞的客戶端權重，繼續聚合
  - 權重差異裁剪，防止極端值
- ✅ **複雜度分析**: 自動計算並顯示空間與通訊複雜度

**配置範例** (`src/env.sh`):
```bash
export SERVER_ALG="fedyoga"
export SERVER_FEDYOGA_HISTORY_WINDOW=5
export SERVER_FEDYOGA_PCA_DIM=4
export SERVER_FEDYOGA_CLIP_THRESHOLD=10.0
```

---

## 🛡️ 錯誤處理與穩定性

### NaN/Inf 自動修復
本框架具備智能錯誤處理機制：

1. **BatchNorm 統計量修復**: 自動重置損壞的 `running_mean`, `running_var`
2. **Critical 參數檢測**: 跳過具有 NaN/Inf 權重的客戶端
3. **詳細診斷訊息**: 提供可能原因和建議解決方案

### 動態端口分配
避免多客戶端並行訓練時的 NCCL port 衝突：
- 自動尋找可用端口 (10000-60000)
- 每個客戶端使用獨立端口

### 實驗日誌
所有輸出自動記錄到 `experiments/{EXP_ID}/orchestrator.log`，方便事後分析。

---

**最後更新**：2025-10-20  
**版本**：v3.0 (穩定性增強版)  
**維護者**：nchc/waue0920

---

## 📸 執行結果快照 (Execution Result Snapshot)

以下是聯邦學習過程中的一些視覺化結果，包含模型驗證成效與訓練指標。

### 1. 模型驗證成果 (Validation Result)
這是使用 Cityscapes 資料集進行 4 個客戶端、5 輪聯邦學習後，對驗證圖片進行物件偵測的結果。
![Validation on Cityscapes](pics/cityscape_c4_r5_val.jpg)

### 2. 訓練指標 (依聯邦輪次)
下圖顯示了在 5 個聯邦輪次中，各項指標 (如 mAP50, mAP50-95) 的變化趨勢。
![Metrics by Round](pics/cityscape_c4_r5_e50_byRound.png)

### 3. 訓練指標 (依訓練週期)
下圖將所有客戶端的訓練週期 (Epoch) 連續繪製，展示了模型在整個訓練過程中的學習曲線。
![Metrics by Epoch](pics/cityscape_c4_r5_e50_byEpoch.png)

### 4. Wandb 儀表板
Wandb 提供了詳細的實驗追蹤，下圖是本次實驗在 Wandb 儀表板上的部分截圖。
![Wandb Dashboard](pics/cityscape_c4_r5_e50_Wandb.png)
