#!/bin/bash#
######## README ###############
# * 這個 script 是用來測試單一輪的聯邦學習流程
# * 目前預設是手動執行一輪的 client 訓練及 server 聚合
# * 目前支援的聚合演算法有: fedavg, fedprox, fedavgm, fedopt, fedawa, fednova
# * 執行方法：
#    a. 找到一個以前已經訓練過的實驗(EXP_ID)
#    b. 執行 run_unit_test.sh
# * 腳本的測試流程是:
#    1. 用 第一輪的 client 的輸出做第一輪的 server 聚合
#    2. 用 第一輪聚合 做 第二輪 client 訓練
# * 觀察重點：
#    1. 訓練時 是否loss 有正常值
###################################################
## 超參數
export EXP_ID=11_cocoR100_4C_3R_202509221628
# ALG="fedavg" # works
# ALG="fedavgm" # works
# ALG="fedawa" # works
# ALG="fednova" # works
ALG="fedopt" # works
# ALG="fedprox" # works

##  main
source ~/fl_gyolo_slurm/src/env.sh
export WROOT=/home/waue0920/fl_gyolo_slurm

## 第一輪 Server 聚合
cd ~/fl_gyolo_slurm/src; python3 server_fedagg.py --input-dir /home/waue0920/fl_gyolo_slurm/experiments/${EXP_ID}/client_outputs/${EXP_ID} --output-file /home/waue0920/fl_gyolo_slurm/experiments/${EXP_ID}/aggregated_weights/w_s_r1.pt --expected-clients 4 --round 1 --algorithm ${ALG};

## 用 第一輪 Server 聚合 做 第二輪 client 訓練
cd ~/fl_gyolo_slurm/gyolo; torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=9527 ./caption/train.py --weights /home/waue0920/fl_gyolo_slurm/experiments/${EXP_ID}/aggregated_weights/w_s_r1.pt --data /home/waue0920/fl_gyolo_slurm/federated_data/cocoR100_4/c1.yaml --cfg /home/waue0920/fl_gyolo_slurm/gyolo/models/caption/gyolo.yaml --project /home/waue0920/fl_gyolo_slurm/experiments/${EXP_ID}/client_outputs/${EXP_ID} --name r2_c1 --device 0 --epochs 2 --batch 4 --img 640 --workers 8 --hyp /home/waue0920/fl_gyolo_slurm/gyolo/data/hyps/hyp.scratch-cap.yaml --optimizer AdamW --flat-cos-lr --no-overlap --close-mosaic 2 --save-period 1 --noplots
