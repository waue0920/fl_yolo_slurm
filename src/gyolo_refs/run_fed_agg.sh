#!/bin/bash
######### README ###############
# * 用來測試每個聚合演算法每一輪是否出錯
#    - 可控制輪數
#    - 可控制演算法 fedavg, fedprox, fedavgm, fedopt, fedawa, fednova
# * 執行方法：
#    a. 找到一個以前已經訓練過的實驗(EXP_ID)
#    b. 執行 run_fed_agg.sh
# * 腳本的測試流程是:
#    1. 第一層迴圈為演算法
#    2. 第二層迴圈為輪數
#    3. 每一輪都用該輪的 client 輸出做演算法的聚合
# * 觀察重點：
#    1. 執行聚合時是否有錯誤訊息
###################################################

###### 可修改參數區 ######
export EXP_ID=11_cocoR100_4C_3R_202509221628
ROUNDS=(1) # (1 2 3)
ALGORITHMS=("fedopt" "fedawa" "fedavgm" "fednova" "fedprox" "fedavg") # "fedopt" "fedawa" "fedavgm" "fednova" "fedprox" "fedavg"
######


cd ~/fl_gyolo_slurm/
source ./src/env.sh

export CLIENT_NUM=4

BASE_DIR="/home/waue0920/fl_gyolo_slurm/experiments/${EXP_ID}"
INPUT_DIR="${BASE_DIR}/client_outputs/${EXP_ID}"
OUTPUT_DIR="${BASE_DIR}/aggregated_weights"



for alg in "${ALGORITHMS[@]}"; do
  for rd in "${ROUNDS[@]}"; do
    OUTPUT_FILE="${OUTPUT_DIR}/w_s_r${rd}.pt"
    python3 ./src/server_fedagg.py \
      --input-dir "${INPUT_DIR}" \
      --output-file "${OUTPUT_FILE}" \
      --expected-clients ${CLIENT_NUM} \
      --round ${rd} \
      --algorithm ${alg}
  done
done
