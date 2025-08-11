# 監控與偵錯指南

本文件提供在執行聯邦學習實驗時，進行系統監控、日誌檢查及故障排除的相關指引。

## 1. Slurm 作業監控

當您提交實驗後，可以使用標準的 Slurm 指令來監控作業狀態。

```bash
# 查看您當前所有作業的狀態 (排隊中、執行中)
squeue -u $USER

# 查看特定作業的詳細資訊 (例如資源分配、執行時間)
scontrol show job <JOB_ID>

# 取消一個正在執行或排隊中的作業
scancel <JOB_ID>
```

## 2. 日誌檢查

日誌是偵錯最重要的資訊來源。本系統的日誌分散在實驗目錄中，各司其職。

```bash
# 1. 查看主協調腳本 (orchestrate.sh) 的完整流程日誌
tail -f experiments/{EXP_ID}/orchestrator.log

# 2. 查看特定客戶端 (例如 client 1, round 1) 的 Slurm 輸出日誌
#    這包含了 YOLOv9 訓練時的詳細畫面輸出
tail -f experiments/{EXP_ID}/slurm_logs/client_1_round_1.out

# 3. 查看聯邦平均 (server_fedavg.py) 的執行日誌
tail -f experiments/{EXP_ID}/fed_avg_logs/round_1.out
```

## 3. 關鍵路徑與檔案檢查

當遇到問題時，可以手動檢查以下關鍵路徑的產出是否符合預期。

```bash
# 檢查資料集是否已成功為客戶端分割
# 應能看到 c1.yaml, c1/, c2.yaml, c2/ ... 等檔案
ls -la federated_data/{DATASET_NAME}_{CLIENT_NUM}/

# 檢查各輪次聚合後的權重檔案是否已生成
ls -la experiments/{EXP_ID}/aggregated_weights/

# 檢查特定客戶端的訓練輸出，特別是權重檔案 best.pt 是否存在
ls -la experiments/{EXP_ID}/client_outputs/round_1/client_*/weights/
```

## 4. 常見錯誤與解決方案

### 錯誤一：環境變數未設定

- **錯誤訊息**: `Error: WROOT environment variable is not set`
- **發生原因**: 在手動執行某些輔助腳本 (如 `client_train.sh`) 時，沒有預先設定好專案根目錄的環境變數。
- **解決方案**: 在執行腳本前，手動匯出必要的環境變數。
  ```bash
  # 設定 WROOT 為您專案的絕對路徑
  export WROOT=$(pwd)
  
  # 檢查是否設定成功
  echo $WROOT
  ```
  > **注意**: 使用主腳本 `orchestrate.sh` 時通常不會遇到此問題，因為它會自動處理。

### 錯誤二：Singularity 映像檔路徑錯誤

- **錯誤訊息**: `Error: Singularity image not found`
- **發生原因**: 腳本在預設路徑 (`${WROOT}/yolo9t2_ngc2306_20241226.sif`) 找不到容器映像檔。
- **解決方案**: 確認您的 `.sif` 檔案確實存在於專案根目錄下。
  ```bash
  ls -la ${WROOT}/yolo9t2_ngc2306_20241226.sif
  ```

### 偵錯技巧：使用 `bash -x` 模式

當您使用手動模式產生的 `sop.sh` 腳本時，若想看到每個指令在執行前的詳細內容 (變數會被代換成實際值)，可以使用 `bash -x` 來執行。這對於追蹤變數設定或路徑問題非常有用。

```bash
# 先產生 SOP 腳本
./orchestrate.sh kitti 4 2 --manual > debug_sop.sh

# 使用 -x 模式執行，畫面上會印出詳細的指令執行過程
bash -x debug_sop.sh
```
