# 模型驗證說明

本文件詳細說明如何使用系統內建的模型驗證功能。

## 功能概述

本系統的驗證功能 (`validate_federated_model.py`) 提供對聯邦學習模型效能的完整分析，支援：

- ✅ **多模型比較**：自動比較基線模型 (baseline) 與實驗中各輪次產出的聚合模型。
- ✅ **詳細指標分析**：提供 mAP@0.5、mAP@0.5:0.95、精確率 (Precision) 和召回率 (Recall)。
- ✅ **每類別分析**：顯示資料集中每個類別的 AP (Average Precision) 值，便於觀察模型對特定類別的學習成效。
- ✅ **資料集分析**：檢測資料集的類別分佈，並標示出模型預測中缺失的類別。
- ✅ **效能比較**：量化計算每個輪次的模型相對於基線模型的效能改善幅度 (百分比)。

## 如何啟用驗證

### 自動化流程中的驗證
在執行 `orchestrate.sh` 自動化腳本時，只需加上 `--val` 旗標，即可在所有聯邦學習輪次結束後，自動觸發完整的驗證流程。

```bash
# 範例：執行一個 4 客戶端、2 輪次的 kitti 實驗，並在結束後進行驗證
./src/orchestrate.sh kitti 4 2 --val
```

### 在手動模式 (SOP) 中包含驗證
若您希望產生的手動執行腳本 (`sop.sh`) 包含驗證步驟，請在產生指令中同樣加上 `--val` 旗標。

```bash
# 範例：產生一個包含驗證步驟的 SOP 腳本
./src/orchestrate.sh kitti 4 2 --manual --val > sop.sh
```

## 手動執行驗證

您也可以隨時針對任何一個已完成的實驗，獨立執行驗證腳本。

### 基本語法
```bash
python3 src/validate_federated_model.py \
    --experiment-dir <實驗結果的路徑> \
    --data-config <資料集設定檔的路徑>
```

### 範例
```bash
# 驗證指定實驗中的所有模型 (基線 + 所有輪次)
```bash
python3 src/validate_federated_model.py \
    --experiment-dir experiments/9_kitti_fedavg_4C_2R_202508051443 \
    --data-config data/kitti.yaml
```
```

### 自訂選項
您也可以指定基線模型或輸出目錄，以進行更靈活的比較。
```bash
# 使用自訂的基線模型和輸出目錄進行驗證
python3 src/validate_federated_model.py \
    --experiment-dir experiments/9_kitti_fedavg_4C_2R_202508051443 \
    --data-config data/cityscapes.yaml \
    --baseline-model /path/to/your/custom_baseline.pt \
    --output-dir /path/to/your/custom_validation_results
```

## 驗證結果說明

驗證完成後，所有結果將儲存在實驗目錄下的 `validation/` 資料夾中 (`<實驗路徑>/validation/`)

### 輸出檔案結構
- **`validation_summary.txt`**: 人類可讀的純文字摘要報告，包含所有重要的比較表格。
- **`model_comparison.json`**: JSON 格式的完整數據，便於程式後續處理或繪圖。
- **`validation_{model_name}/`**: 每個模型 (基線、round_1 等) 的獨立驗證結果資料夾，包含由 YOLOv9 產生的詳細驗證日誌和圖表。

### 如何解讀 `validation_summary.txt`
這份報告是快速了解模型效能的關鍵。

**報告範例：**
```
### Federated Learning Model Validation Summary ###

# 1. 整體指標比較表
Model           mAP@0.5    mAP@0.75   mAP@0.5:0.95 Precision  Recall    
--------------------------------------------------------------------------------
baseline        0.752      0.543      0.481        0.743      0.687     
round_1         0.768      0.558      0.495        0.751      0.702     
round_2         0.784      0.572      0.508        0.762      0.718     

# 2. 效能改善分析
=== Performance Analysis ===
Baseline Model (baseline):
  mAP@0.5 = 0.752
  mAP@0.5:0.95 = 0.481

round_2:
  mAP@0.5 = 0.784 (Δ = +0.032, +4.3%)
  mAP@0.5:0.95 = 0.508 (Δ = +0.027, +5.6%)
```
從範例中可以清楚看到，經過兩輪聯邦學習後，`mAP@0.5` 相比基線模型提升了 4.3%。

