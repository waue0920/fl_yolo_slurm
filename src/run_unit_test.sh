#!/bin/bash
#
######## README ###############
# * 這個 script 是用來測試單一輪的聯邦學習流程
# * 目前預設是手動執行一輪的 client 訓練及 server 聚合
# * 目前支援的聚合演算法有: fedavg, fedprox, fedavgm, fedopt, fedyoga, fednova
# * 執行方法：
#    a. 找到一個以前已經訓練過的實驗(EXP_ID)
#    b. 修改下面的 EXP_ID 和 ALG 參數
#    c. 執行 ./src/run_unit_test.sh
# * 腳本的測試流程是:
#    1. 用 第一輪的 client 的輸出做第一輪的 server 聚合
#    2. 用 第一輪聚合 做 第二輪 client 訓練
# * 觀察重點：
#    1. 訓練時 是否loss 有正常值
#    2. 聚合是否成功
#    3. NaN/Inf 處理是否正常
###################################################

## 超參數設定
export EXP_ID=11_kitti_4C_3R_202509221628  # 修改為您的實驗ID
# ALG="fedavg"   # works
# ALG="fedavgm"  # works
ALG="fedyoga"    # works - 測試新版 FedYOGA
# ALG="fednova"  # works
# ALG="fedopt"   # works
# ALG="fedprox"  # works

##  載入環境設定
source ~/fl_yolo_slurm/src/env.sh
export WROOT=/home/waue0920/fl_yolo_slurm

echo "=========================================="
echo "  YOLOv9 Federated Learning Unit Test"
echo "=========================================="
echo "Experiment ID: ${EXP_ID}"
echo "Algorithm:     ${ALG}"
echo "Working Dir:   ${WROOT}"
echo "=========================================="
echo ""

## 第一輪 Server 聚合測試
echo "=== Step 1: Testing Server Aggregation (Round 1) ==="
cd ~/fl_yolo_slurm/src

python3 server_fedagg.py \
    --input-dir /home/waue0920/fl_yolo_slurm/experiments/${EXP_ID}/client_outputs/${EXP_ID} \
    --output-file /home/waue0920/fl_yolo_slurm/experiments/${EXP_ID}/aggregated_weights/w_s_r1_test.pt \
    --expected-clients 4 \
    --round 1 \
    --algorithm ${ALG}

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Server aggregation FAILED"
    exit 1
else
    echo ""
    echo "✅ Server aggregation PASSED"
fi

echo ""
echo "=== Step 2: Testing Client Training with Aggregated Weights (Round 2) ==="
## 用 第一輪 Server 聚合 做 第二輪 client 訓練
cd ~/fl_yolo_slurm/yolov9

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=9527 \
    ./train_dual.py \
    --weights /home/waue0920/fl_yolo_slurm/experiments/${EXP_ID}/aggregated_weights/w_s_r1_test.pt \
    --data /home/waue0920/fl_yolo_slurm/federated_data/kitti_4/c1.yaml \
    --cfg /home/waue0920/fl_yolo_slurm/yolov9/models/detect/yolov9-c.yaml \
    --project /home/waue0920/fl_yolo_slurm/experiments/${EXP_ID}/client_outputs/${EXP_ID} \
    --name r2_c1_test \
    --device 0 \
    --epochs 2 \
    --batch 4 \
    --img 640 \
    --workers 8 \
    --hyp /home/waue0920/fl_yolo_slurm/yolov9/data/hyps/hyp.scratch-high.yaml \
    --optimizer AdamW \
    --flat-cos-lr \
    --no-overlap \
    --close-mosaic 2 \
    --save-period 1 \
    --noplots

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Client training FAILED"
    exit 1
else
    echo ""
    echo "✅ Client training PASSED"
fi

echo ""
echo "=========================================="
echo "  ✅ Unit Test COMPLETED Successfully"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - Aggregated model: experiments/${EXP_ID}/aggregated_weights/w_s_r1_test.pt"
echo "  - Training output:  experiments/${EXP_ID}/client_outputs/${EXP_ID}/r2_c1_test/"
echo ""
