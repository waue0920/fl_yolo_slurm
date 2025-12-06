#####################
## Global Project and Experiment Parameters
#####################
export WROOT="/home/waue0920/fl_yolo_slurm"
export EXPERIMENTS_BASE_DIR="${WROOT}/experiments"

## Dataset
export DATASET_NAME="kittiO" #bdd100k, kittiO, kitti, sim10k, foggy, cityscapes | kittiOA010 ...
export CLIENT_NUM=10   # Number of Clients 
# Requires ${DATASET_NAME}_${CLIENT_NUM} folder in $WROOT/federated_data/

## Environment
export SINGULARITY_IMG="${WROOT}/yolo9t2_ngc2306_20241226.sif"
export INITIAL_WEIGHTS="yolov9-c.pt"  # yolov9-c.pt or set to "" to train from scratch

#####################
## Slurm 
#####################
export SLURM_PARTITION=gp4d
export SLURM_ACCOUNT="GOV113119" # 

#####################
## FL Client Slurm Parameters
#####################
## FL
export TOTAL_ROUNDS=12  # FL Rounds
export EPOCHS=10 # 
export FL_HYP_THRESHOLD=11 # FedYOGA local train, adjust with ROUND and EPOCHS
## yolo
export BATCH_SIZE=32   # Must be multiple of GPU count: usually GPUsx8 or high GPUsx16 
export WORKER=8   # cpu = gpu x 4
export IMG_SIZE=640
export HYP="${WROOT}/yolov9/data/hyps/hyp.scratch-high.yaml" # hyp.scratch-high.yaml or hyp.fedyoga.yaml
export FedyogaHYP="${WROOT}/yolov9/data/hyps/hyp.fedyoga.yaml"
export MODEL_CFG="${WROOT}/yolov9/models/detect/yolov9-c.yaml" # 
# Dynamic parameter format, easy to reference and override
export TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${HYP} --close-mosaic 15"
### (low) export TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${HYP} --optimizer AdamW --flat-cos-lr --close-mosaic 2 --save-period 1 --noplots"
## Parallel
#export CLIENT_NODES=1  # Currently each Client only supports single-node multi-GPU, due to NCCL port conflict
export CLIENT_GPUS=2
export CLIENT_CPUS=8   # cpu = gpu x 4

#####################
## FL Server Parameters
#####################
export SERVER_ALG="fedawa"   # Supports fedavg, fedprox, fedavgm, fedopt, fedyoga, fedawa, fednova

# FedProx Hyperparameters (Server-side naming)
export SERVER_FEDPROX_MU=0.01  # Proximal term coefficient for FedProx
# FedAvgM Hyperparameters (Server-side naming)
export SERVER_FEDAVGM_LR=1.0  # FedAvgM server learning rate
export SERVER_FEDAVGM_MOMENTUM=0.9 # FedAvgM server momentum
# FedAWA Hyperparameters (Server-side naming, based on original paper)
export SERVER_FEDAWA_SERVER_EPOCHS=1  # Epochs for T_weights optimization
export SERVER_FEDAWA_SERVER_OPTIMIZER="adam"  # Optimizer: adam or sgd
export SERVER_FEDAWA_SERVER_LR=0.001  # Server LR (adam:0.001, sgd:0.01, auto-selected)
export SERVER_FEDAWA_GAMMA=1.0  # Aggregation weight scaling factor
export SERVER_FEDAWA_REG_DISTANCE="cos"  # Distance metric: cos (Cosine) or euc (Euclidean)

# FedYOGA Hyperparameters
export SERVER_FEDYOGA_SERVER_EPOCHS=1  # Epochs for T_weights optimization
export SERVER_FEDYOGA_SERVER_LR=0.001  # Server LR
export SERVER_FEDYOGA_GAMMA=1.0  # Aggregation weight scaling factor
export SERVER_FEDYOGA_LAYER_GROUP_SIZE=1  # Layers per group (1=layer-wise, >1=grouped)


# FedOpt Hyperparameters (Server-side naming)
export SERVER_FEDOPT_LR=0.001
export SERVER_FEDOPT_BETA1=0.9
export SERVER_FEDOPT_BETA2=0.999
export SERVER_FEDOPT_EPS=1e-8
# FedNova Hyperparameters (Server-side naming)
export SERVER_FEDNOVA_MU=0.0
export SERVER_FEDNOVA_LR=1.0



# Currently scaffold client local control_variate and server control_variate not implemented
