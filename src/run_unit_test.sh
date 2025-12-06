#!/bin/bash
#
######## README ###############
# * This script calculates a single round of FL process test
# * Default is manual execution of one round client training and server aggregation
# * Supported algorithms: fedavg, fedprox, fedavgm, fedopt, fedyoga, fednova
# * Execution:
#    a. Find a previous experiment ID (EXP_ID)
#    b. Modify EXP_ID and ALG parameters below
#    c. Run ./src/run_unit_test.sh
# * Test flow:
#    1. Use Round 1 client output for Round 1 server aggregation
#    2. Use Round 1 aggregation result for Round 2 client training
# * Key Observations:
#    1. Is loss normal during training?
#    2. Is aggregation successful?
#    3. Is NaN/Inf handling working?
###################################################

## Hyperparameter Settings
export EXP_ID=11_kitti_4C_3R_202509221628  # Modify to your Experiment ID
# ALG="fedavg"   # works
# ALG="fedavgm"  # works
ALG="fedyoga"    # works - Testing new FedYOGA
# ALG="fednova"  # works
# ALG="fedopt"   # works
# ALG="fedprox"  # works

##  Load environment settings
source ./src/env.sh
# WROOT is exported by env.sh

echo "=========================================="
echo "  YOLOv9 Federated Learning Unit Test"
echo "=========================================="
echo "Experiment ID: ${EXP_ID}"
echo "Algorithm:     ${ALG}"
echo "Working Dir:   ${WROOT}"
echo "=========================================="
echo ""

## Round 1 Server Aggregation Test
echo "=== Step 1: Testing Server Aggregation (Round 1) ==="
cd ${WROOT}/src

python3 server_fedagg.py \
    --input-dir ${WROOT}/experiments/${EXP_ID}/client_outputs/${EXP_ID} \
    --output-file ${WROOT}/experiments/${EXP_ID}/aggregated_weights/w_s_r1_test.pt \
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
## Use Round 1 Server Aggregation for Round 2 Client Training
cd ${WROOT}/yolov9

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=9527 \
    ./train_dual.py \
    --weights ${WROOT}/experiments/${EXP_ID}/aggregated_weights/w_s_r1_test.pt \
    --data ${WROOT}/federated_data/kitti_4/c1.yaml \
    --cfg ${WROOT}/yolov9/models/detect/yolov9-c.yaml \
    --project ${WROOT}/experiments/${EXP_ID}/client_outputs/${EXP_ID} \
    --name r2_c1_test \
    --device 0 \
    --epochs 2 \
    --batch 4 \
    --img 640 \
    --workers 8 \
    --hyp ${WROOT}/yolov9/data/hyps/hyp.scratch-high.yaml \
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
