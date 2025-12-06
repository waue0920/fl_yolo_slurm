#####################
## Global Project and Experiment Parameters
#####################
# Get the directory where env.sh is located (src/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# WROOT is the parent directory of SCRIPT_DIR
export WROOT="$(dirname "$SCRIPT_DIR")"
export EXPERIMENTS_BASE_DIR="${WROOT}/experiments"

## Dataset
export DATASET_NAME="${DATASET_NAME:-kittiO}" #bdd100k, kittiO, kitti, sim10k, foggy, cityscapes | kittiOA010 ...
export CLIENT_NUM=4   # Number of Clients 
# Requires ${DATASET_NAME}_${CLIENT_NUM} folder in $WROOT/federated_data/

## Environment
# export SINGULARITY_IMG="${WROOT}/yolo9t2_ngc2306_20241226.sif"
export INITIAL_WEIGHTS=""  # yolov7x.pt or set to "" to train from scratch

#####################
## Slurm 
#####################
# export SLURM_PARTITION=gp4d
# export SLURM_ACCOUNT="GOV113119" # 

#####################
## FL client 端的 slurm 參數
#####################
## FL
export TOTAL_ROUNDS=6  # FL Rounds
export EPOCHS=10 # 
# export FL_HYP_THRESHOLD=5 # FedYOGA local train, adjust with ROUND and EPOCHS
## yolo
export BATCH_SIZE=32   # Must be multiple of GPU count: usually GPUsx8 or high GPUsx16 
export WORKER=8   # cpu = gpu x 4
export IMG_SIZE=640
# export HYP="${WROOT}/yolov9/data/hyps/hyp.scratch-high.yaml" # hyp.scratch-high.yaml or hyp.fedyoga.yaml
# export FedyogaHYP="${WROOT}/yolov9/data/hyps/hyp.fedyoga.yaml"
# export MODEL_CFG="${WROOT}/yolov9/models/detect/yolov9-c.yaml" # 
# Dynamic parameter format, easy to reference and override
export HYP="${WROOT}/yolov7/data/hyp.scratch.p5.yaml"
export MODEL_CFG="${WROOT}/yolov7/cfg/training/yolov7-kitti.yaml"
export TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${HYP}"
### (low) export TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${HYP} --optimizer AdamW --flat-cos-lr --save-period 1 --noplots"
## Parallel
#export CLIENT_NODES=1  # Currently each Client only supports single-node multi-GPU, due to NCCL port conflict
export CLIENT_GPUS=2
export CLIENT_CPUS=8   # cpu = gpu x 4

#####################
## FL Server Parameters
#####################
export SERVER_ALG="${SERVER_ALG:-fedawa}"   # 支持 fedavg, fedprox, fedavgm, fedopt, fedyoga, fedawa, fednova

# FedProx Hyperparameters
export SERVER_FEDPROX_MU=0.01  # Proximal term coefficient
# FedAvgM Hyperparameters
export SERVER_FEDAVGM_LR=1.0  # Server LR
export SERVER_FEDAVGM_MOMENTUM=0.9 # Server Momentum
# FedAWA Hyperparameters
export SERVER_FEDAWA_SERVER_EPOCHS=1  # T_weights epochs
export SERVER_FEDAWA_SERVER_OPTIMIZER="adam"  # Optimizer: adam or sgd
export SERVER_FEDAWA_SERVER_LR=0.001  # Server LR
export SERVER_FEDAWA_GAMMA=1.0  # Scaling factor
export SERVER_FEDAWA_REG_DISTANCE="cos"  # Distance metric: cos or euc

# FedYOGA Hyperparameters
export SERVER_FEDYOGA_SERVER_EPOCHS=1  # T_weights epochs
export SERVER_FEDYOGA_SERVER_LR=0.001  # Server LR
export SERVER_FEDYOGA_GAMMA=1.0  # Scaling factor
export SERVER_FEDYOGA_LAYER_GROUP_SIZE=1  # Layers per group (1=layer-wise, >1=grouped)


# FedOpt Hyperparameters
export SERVER_FEDOPT_LR=0.001
export SERVER_FEDOPT_BETA1=0.9
export SERVER_FEDOPT_BETA2=0.999
export SERVER_FEDOPT_EPS=1e-8
# FedNova Hyperparameters
export SERVER_FEDNOVA_MU=0.0
export SERVER_FEDNOVA_LR=1.0



# Currently scaffold not implemented
