
#####################
## 全域專案與實驗參數
#####################
export WROOT="/home/waue0920/fl_yolo_slurm"
export EXPERIMENTS_BASE_DIR="${WROOT}/experiments"

## Dataset
export DATASET_NAME="kittiA010" # 
export CLIENT_NUM=4   # Client 端數量
# 以上在  $WROOT/federated_data/ 內要有 ${DATASET_NAME}_${CLIENT_NUM} 的資料夾

## Environment
export SINGULARITY_IMG="${WROOT}/yolo9t2_ngc2306_20241226.sif"
export INITIAL_WEIGHTS="yolov9-c.pt"  # yolov9-c.pt 或設為 "" 代表從頭訓練

#####################
## Slurm 
#####################
export SLURM_PARTITION=gp4d
export SLURM_ACCOUNT="GOV109134" # 

#####################
## FL client 端的 slurm 參數
#####################
## FL
export TOTAL_ROUNDS=20  # FL Rounds
export EPOCHS=5
## yolo
export BATCH_SIZE=16   # 需要是 gpu 數量的n數: 一般 GPUsx8 高 GPUsx16 
export WORKER=8   # cpu = gpu x 4
export IMG_SIZE=640
export HYP="${WROOT}/yolov9/data/hyps/hyp.scratch-high.yaml" # hyp.scratch-cap.yaml or hyp.scratch-cap-e.yaml
export MODEL_CFG="${WROOT}/yolov9/models/detect/yolov9-c.yaml" # 
# 動態參數形式，方便引用和覆蓋
export TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${HYP} --optimizer AdamW --flat-cos-lr --close-mosaic 2 --save-period 1 --noplots"
## Parallel
#export CLIENT_NODES=1  # 目前每個Client只支援單節點多GPU運算，因為NCCL port 會衝突
export CLIENT_GPUS=2
export CLIENT_CPUS=8   # cpu = gpu x 4

#####################
## FL Server 端的參數
#####################
export SERVER_ALG="fedavg"   # 支持 fedavg, fedprox, fedavgm, fedopt, fedyoga, fednova

# FedProx 演算法超參數（Server端命名
export SERVER_FEDPROX_MU=0.01  # FedProx 的 proximal term 係數
# FedAvgM 演算法超參數（Server端命名
export SERVER_FEDAVGM_LR=1.0  # FedAvgM 的 server learning rate
export SERVER_FEDAVGM_MOMENTUM=0.9 # FedAvgM 的 server momentum
# FedYOGA 演算法超參數（Server端命名）
export SERVER_FEDYOGA_HISTORY_WINDOW=5
export SERVER_FEDYOGA_PCA_DIM=4
export SERVER_FEDYOGA_SOFTMAX_TEMPERATURE=1.0
export SERVER_FEDYOGA_LOSSDROP_WEIGHT=1.0
export SERVER_FEDYOGA_GRADVAR_WEIGHT=1.0
export SERVER_FEDYOGA_PCA_SOLVER="full"
export SERVER_FEDYOGA_NORM_EPS=1e-8
export SERVER_FEDYOGA_CLIP_THRESHOLD=10.0  # Non-IID 數值穩定性：權重差異裁剪閾值
# FedOpt 演算法超參數（Server端命名）
export SERVER_FEDOPT_LR=0.001
export SERVER_FEDOPT_BETA1=0.9
export SERVER_FEDOPT_BETA2=0.999
export SERVER_FEDOPT_EPS=1e-8
# FedNova 演算法超參數（Server端命名）
export SERVER_FEDNOVA_MU=0.0
export SERVER_FEDNOVA_LR=1.0

# 目前 scaffold 尚未實作 client 端的 local control_variate 與 server control_variate
