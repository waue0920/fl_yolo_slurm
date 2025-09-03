
#####################
## 全域專案與實驗參數
#####################
export WROOT="/home/waue0920/fl_yolo_slurm"
export EXPERIMENTS_BASE_DIR="${WROOT}/experiments"

## Dataset
export DATASET_NAME="sim10k"
## Environment
export SINGULARITY_IMG="${WROOT}/yolo9t2_ngc2306_20241226.sif"

#####################
## Slurm 
#####################
export SLURM_PARTITION=gp2d
export SLURM_ACCOUNT="GOV113038"


#####################
## FL client 端的 slurm 參數
#####################
## FL
export CLIENT_NUM=4   # Client 端數量
export TOTAL_ROUNDS=3  # FL Rounds
export EPOCHS=30
## Yolo
export BATCH_SIZE=16
export IMG_SIZE=640

## Parallel
#export CLIENT_NODES=1  # 目前每個Client只支援單節點多GPU運算，因為NCCL port 會衝突
export CLIENT_GPUS=2
export CLIENT_CPUS=8

#####################
## FL Server 端的參數
#####################
export SERVER_ALG="fedprox"   # 支持 fedavg, fedprox, scaffold
export SERVER_FEDPROX_MU=0.01  # FedProx 的 proximal term 係數
