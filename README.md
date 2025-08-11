# YOLOv9 Federated Learning Framework

一個基於 YOLOv9 的聯邦學習系統，使用 Slurm 叢集環境進行分散式訓練和聚合。

## 📋 目錄

- [系統概述](#系統概述)
- [環境需求](#環境需求)
- [安裝設定](#安裝設定)
- [快速開始](#快速開始)
- [詳細使用說明](#詳細使用說明)
- [目錄結構](#目錄結構)
- [腳本說明](#腳本說明)
- [實驗管理](#實驗管理)
- [故障排除](#故障排除)

## 🎯 系統概述


### 聯邦學習流程

```
Round 1: yolov9-c.pt → [Client1, Client2, Client3, Client4] → w_s_r1.pt
Round 2: w_s_r1.pt  → [Client1, Client2, Client3, Client4] → w_s_r2.pt
Round 3: w_s_r2.pt  → [Client1, Client2, Client3, Client4] → w_s_r3.pt
...
```

## 🛠️ 環境需求

### 必要條件
- **作業系統**：Linux 
- **作業調度器**：Slurm Workload Manager
- **容器引擎**：Singularity
- **Python**：≥ 3.8
- **GPU**：NVIDIA GPU (支援 CUDA)

### 軟體準備

- PyTorch ≥ 2.1.0
- Wandb (實驗追蹤)
- YOLOv9 相關依賴 (參見 `yolov9/requirements.txt`)
TODO : 需補充 yolov9 下載網址

## 📦 安裝設定

### 1. 下載專案
```bash
git clone <repository-url>
cd fl_yolo
```

### 2. 準備 Singularity 映像檔
確保 `yolo9t2_ngc2306_20241226.sif` 映像檔存在於專案根目錄：
```bash
ls -la yolo9t2_ngc2306_20241226.sif
```
TODO : 需補充 singualrity 容器 下載網址

### 3. 準備資料集
將資料集放置於對應目錄：
```
datasets/
├── kitti/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── ...

data/
├── kitti.yaml
└── ...
```
TODO : 需補充說明 dataset 下載網址


### 4. 初始權重
確保 YOLOv9 預訓練權重存在：
```bash
ls -la yolov9-c.pt
```
TODO : 需補充 yolov9-c.pt 下載網址

## 🚀 快速開始

### 自動模式 (推薦)
執行完整的聯邦學習實驗：
```bash
cd src
./orchestrate.sh kitti 4 2
```

### 手動模式
生成標準操作程序 (SOP)：
```bash
cd src
./orchestrate.sh kitti 4 2 --manual > sop.sh
```
然後按照 `sop.sh` 中的命令逐步執行。

## 📚 詳細使用說明

### 基本語法
```bash
./orchestrate.sh <DATASET_NAME> <CLIENT_NUM> <TOTAL_ROUNDS>
./orchestrate.sh <DATASET_NAME> <CLIENT_NUM> <TOTAL_ROUNDS> [--manual] [--val]
```

### 參數說明
- `DATASET_NAME`: 資料集名稱 (例如：kitti, cityscapes)
- `CLIENT_NUM`: 客戶端數量 (例如：4)
- `TOTAL_ROUNDS`: 聯邦學習輪次 (例如：2, 5)
- `--manual`: 可選，生成手動執行的 SOP
- `--val`: 可選，包含模型驗證步驟

### 範例
```bash
# 4個客戶端，進行5輪聯邦學習
./orchestrate.sh kitti 4 5

# 包含驗證的聯邦學習實驗
./orchestrate.sh kitti 4 2 --val

# 8個客戶端，進行3輪聯邦學習，生成手動執行文檔（含驗證）
./orchestrate.sh cityscapes 8 3 --manual --val > cityscapes_sop.sh
```

## 📊 模型驗證功能

### 驗證功能概述
本系統提供完整的聯邦學習模型驗證功能，支援：
- ✅ **多模型比較**：自動比較基線模型與各輪次聚合模型
- ✅ **詳細指標分析**：mAP@0.5、mAP@0.5:0.95、精度、召回率
- ✅ **每類別分析**：顯示各類別的 AP 值和改善情況
- ✅ **資料集分析**：檢測資料集類別分布和缺失類別
- ✅ **效能比較**：計算各輪次相對於基線的改善幅度

### 啟用驗證
在 orchestrate.sh 命令中加入 `--val` 參數：

```bash
# 自動模式（含驗證）
./orchestrate.sh kitti 4 2 --val

# 手動模式（含驗證）
./orchestrate.sh kitti 4 2 --manual --val > sop.sh
```

### 手動執行驗證
您也可以獨立執行驗證腳本：

```bash
# 驗證指定實驗的所有模型
python3 src/validate_federated_model.py \
    --experiment-dir experiments/9_kitti_4C_2R_202508051443 \
    --data-config data/kitti.yaml

# 自訂基線模型和輸出目錄
python3 src/validate_federated_model.py \
    --experiment-dir experiments/9_kitti_4C_2R_202508051443 \
    --data-config data/cityscapes.yaml \
    --baseline-model custom_baseline.pt \
    --output-dir custom_validation_results
```

### 驗證結果說明
驗證完成後會在 `${EXP_DIR}/validation/` 目錄產生：

1. **validation_summary.txt** - 人類可讀的摘要報告
   - 整體指標比較表
   - 每類別 mAP 分析表
   - 效能改善分析
   - 資料集類別分布分析

2. **model_comparison.json** - 詳細的 JSON 格式結果
   - 完整的指標數據
   - 每個模型的詳細指標
   - 每類別的 AP 值

3. **validation_{model_name}/** - 各模型的詳細驗證結果
   - 每個模型的獨立指標檔案

### 驗證報告範例
```
### Federated Learning Model Validation Summary ###

Model           mAP@0.5    mAP@0.75   mAP@0.5:0.95 Precision  Recall    
--------------------------------------------------------------------------------
baseline        0.752      0.543      0.481        0.743      0.687     
round_1         0.768      0.558      0.495        0.751      0.702     
round_2         0.784      0.572      0.508        0.762      0.718     

=== Performance Analysis ===
Baseline Model (baseline):
  mAP@0.5 = 0.752
  mAP@0.5:0.95 = 0.481

round_2:
  mAP@0.5 = 0.784 (Δ = +0.032, +4.3%)
  mAP@0.5:0.95 = 0.508 (Δ = +0.027, +5.6%)
```

# 8個客戶端，進行3輪聯邦學習，生成手動執行文檔
```
./orchestrate.sh cityscapes 8 3 --manual > cityscapes_sop.sh
```

## 📁 目錄結構

### 專案結構
```
fl_yolo/
├── README.md                          # 本文檔
├── psudocode.txt                       # 系統設計文檔
├── yolo9t2_ngc2306_20241226.sif        # Singularity 容器映像檔
├── yolov9-c.pt                         # YOLOv9 預訓練權重
├── src/                                # 主要腳本目錄
│   ├── orchestrate.sh                  # 主協調腳本
│   ├── data_prepare.py                 # 資料分割腳本
│   ├── fl_client_train.sh              # 客戶端批次訓練調度器
│   ├── client_train.sh                 # 客戶端訓練執行器
│   ├── client_train.sb                 # 客戶端 Slurm 腳本
│   ├── fl_server_fedavg.sh             # 聯邦平均作業調度器
│   ├── server_fedavg.py                # 聯邦平均演算法
│   ├── server_fedavg.sb                # 聯邦平均 Slurm 腳本
│   └── validate_federated_model.py     # 模型驗證腳本
├── data/                               # 資料集配置檔
│   ├── kitti.yaml
│   └── ...
├── datasets/                           # 原始資料集
│   ├── kitti/
│   └── ...
├── federated_data/                     # 分割後的客戶端資料
│   └── kitti/
│       ├── c1.yaml
│       ├── c1/
│       ├── c2.yaml
│       ├── c2/
│       └── ...
├── experiments/                        # 實驗結果
│   └── {EXP_ID}/
│       ├── orchestrator.log
│       ├── slurm_logs/
│       ├── client_outputs/
│       ├── aggregated_weights/
│       └── fed_avg_logs/
└── yolov9/                            # YOLOv9 原始碼
```

### 實驗輸出結構
```
experiments/{EXP_ID}/
├── orchestrator.log                    # 主協調腳本日誌
├── slurm_logs/                        # Slurm 作業日誌
│   └── ...
├── client_outputs/                    # 客戶端訓練結果
│   ├── round_1/
│   │   ├── client_1/                  # YOLOv9 訓練輸出
│   │   │   ├── weights/
│   │   │   │   └── best.pt           # 最佳權重
│   │   │   ├── results.csv           # 訓練指標
│   │   │   └── ...
│   │   └── ...
│   └── round_2/
├── aggregated_weights/                # 聯邦平均結果
│   ├── w_s_r1.pt                     # 第1輪聚合權重
│   ├── w_s_r2.pt                     # 第2輪聚合權重
│   └── ...
├── fed_avg_logs/                      # 聯邦平均日誌
│   ├── round_1.out
│   ├── round_1.err
│   └── ...
└── validation/                        # 模型驗證結果 (--val 啟用)
    ├── validation_summary.txt         # 人類可讀摘要報告
    ├── model_comparison.json          # 詳細 JSON 結果
    ├── validation_baseline/           # 基線模型驗證結果
    │   └── baseline_metrics.json
    ├── validation_round_1/            # 第1輪模型驗證結果
    │   └── round_1_metrics.json
    └── validation_round_2/            # 第2輪模型驗證結果
        └── round_2_metrics.json
```

## 🔧 腳本說明

### 核心腳本

#### 1. `orchestrate.sh` - 主協調腳本
**功能**：統籌整個聯邦學習流程
**模式**：
- 自動模式：直接執行完整流程
- 手動模式：生成 SOP 文檔

#### 2. `data_prepare.py` - 資料分割
**功能**：將資料集分割為多個客戶端子集
**特點**：
- 自動檢測已分割的資料集
- 使用符號連結節省空間
- 生成客戶端專用配置檔

#### 3. `fl_client_train.sh` - 客戶端調度器
**功能**：批次提交所有客戶端的訓練作業
**特點**：
- 並行提交多個 Slurm 作業
- 自動生成 Wandb 專案和運行名稱
- 處理權重檔案路徑邏輯

#### 4. `server_fedavg.py` - 聯邦平均
**功能**：聚合多個客戶端的模型權重
**演算法**：標準聯邦平均 (FedAvg)
```python
w_global = (1/K) * Σ(w_k)  # K為客戶端數量
```

#### 5. `validate_federated_model.py` - 模型驗證
**功能**：驗證聯邦學習模型效能
**特點**：
- 自動發現實驗中的所有模型（基線 + 各輪次）
- 使用 YOLOv9 核心組件進行驗證
- 生成詳細的指標分析和比較報告
- 包含資料集類別分布分析

**輸出**：
- 整體指標比較（mAP@0.5、mAP@0.5:0.95、精度、召回率）
- 每類別 AP 分析表
- 效能改善分析（相對於基線的提升幅度）
- JSON 格式的詳細結果數據

### 輔助腳本

#### `client_train.sh` - 單一客戶端訓練
**功能**：執行單個客戶端的 YOLOv9 訓練
**特點**：參數化設計，支援不同配置

#### Slurm 腳本 (`.sb`)
**功能**：定義 Slurm 作業的資源需求和執行環境
**特點**：GPU 分配、記憶體管理、容器執行

## 🆔 實驗管理

### 實驗 ID 格式
```
{RUN_NUM}_{DATASET}_{CLIENT_NUM}C_{TOTAL_ROUNDS}R_{TIMESTAMP}_{MODE}
```

### 範例
```
1_kitti_4C_5R_202508051830_manual
│ │     │  │ │              │
│ │     │  │ │              └─ 模式 (manual/auto)
│ │     │  │ └─ 時間戳 (YYYYMMDDHHMM)
│ │     │  └─ 輪次數
│ │     └─ 客戶端數量
│ └─ 資料集名稱
└─ 實驗編號
```

### Wandb 整合
- **專案名稱**：去除時間戳和模式的簡化 ID
- **運行名稱**：`{簡化ID}_c{客戶端}_r{輪次}`
- **範例**：
  - 專案：`1_kitti_4C_5R`
  - 運行：`1_kitti_4C_5R_c1_r1`, `1_kitti_4C_5R_c2_r1`

## 🔍 監控與除錯

### Slurm 作業監控
```bash
# 查看當前作業狀態
squeue -u $USER

# 查看特定作業詳情
scontrol show job <JOB_ID>

# 取消作業
scancel <JOB_ID>
```

### 日誌檢查
```bash
# 查看主協調腳本日誌
tail -f experiments/{EXP_ID}/orchestrator.log

# 查看客戶端訓練日誌
tail -f experiments/{EXP_ID}/slurm_logs/client_1_round_1.out

# 查看聯邦平均日誌
tail -f experiments/{EXP_ID}/fed_avg_logs/round_1.out
```

### 常見問題檢查
```bash
# 檢查資料集是否準備就緒
ls -la federated_data/{DATASET_NAME}/

# 檢查權重檔案
ls -la experiments/{EXP_ID}/aggregated_weights/

# 檢查客戶端輸出
ls -la experiments/{EXP_ID}/client_outputs/round_1/client_*/weights/
```

## 🚨 故障排除

### 環境變數設定
執行任何腳本前，請確保設定必要的環境變數：
```bash
export EXP_ID=<實驗ID>
export WROOT=<專案根目錄絕對路徑>
export DATASET_NAME=<資料集名稱>
export CLIENT_NUM=<客戶端數量>
```

### 常見錯誤解決

#### 1. 環境變數未設定
**錯誤訊息**：`Error: WROOT environment variable is not set`
**解決方案**：在執行任何腳本前設定環境變數
```bash
# 確保 WROOT 環境變數正確設定
echo $WROOT
```

#### 2. Singularity 映像檔路徑錯誤
**錯誤訊息**：`Error: Singularity image not found`
**解決方案**：
```bash
# 確保 singularity 檔存在
ls -la ${WROOT}/yolo9t2_ngc2306_20241226.sif
```

####  3. 手動模式 + 除錯
```bash
# 生成 SOP 後逐步執行
./orchestrate.sh kitti 4 2 --manual > debug_sop.sh
bash -x debug_sop.sh  # 逐行顯示執行過程
```

#### 單獨測試組件
<!-- ```bash
# 測試資料準備
python3 src/data_prepare.py --dataset-name kitti --num-clients 4

# 測試單一客戶端訓練
./src/client_train.sh --data-yaml federated_data/kitti/c1.yaml \
                      --weights-in yolov9-c.pt \
                      --project-out /tmp/test \
                      --name-out test_run

# 測試聯邦平均
python3 src/server_fedavg.py --input-dir /path/to/client_outputs \
                              --output-file /tmp/aggregated.pt \
                              --expected-clients 4
``` -->

---

**最後更新**：2025-08-05  
**版本**：v1.0  
**維護者**：[nchc/waue0920]
