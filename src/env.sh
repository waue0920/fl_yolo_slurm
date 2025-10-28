
#####################
## 全域專案與實驗參數
#####################
export WROOT="/home/waue0920/fl_yolo_slurm"
export EXPERIMENTS_BASE_DIR="${WROOT}/experiments"

## Dataset
export DATASET_NAME="${DATASET_NAME:-kittiO}" #bdd100k, kittiO, kitti, sim10k, foggy, cityscapes | kittiOA010 ...
export CLIENT_NUM=4   # Client 端數量
# 以上在  $WROOT/federated_data/ 內要有 ${DATASET_NAME}_${CLIENT_NUM} 的資料夾

## Environment
export SINGULARITY_IMG="${WROOT}/yolo9t2_ngc2306_20241226.sif"
export INITIAL_WEIGHTS="yolov9-c.pt"  # yolov9-c.pt 或設為 "" 代表從頭訓練

#####################
## Slurm 
#####################
export SLURM_PARTITION=gp4d
export SLURM_ACCOUNT="GOV113119" # 

#####################
## FL client 端的 slurm 參數
#####################
## FL
export TOTAL_ROUNDS=5  # FL Rounds
export EPOCHS=10
## yolo
export BATCH_SIZE=32   # 需要是 gpu 數量的n數: 一般 GPUsx8 高 GPUsx16 
export WORKER=8   # cpu = gpu x 4
export IMG_SIZE=640
export HYP="${WROOT}/yolov9/data/hyps/hyp.scratch-high.yaml" # hyp.scratch-high.yaml or hyp.fedyoga.yaml
export MODEL_CFG="${WROOT}/yolov9/models/detect/yolov9-c.yaml" # 
# 動態參數形式，方便引用和覆蓋
export TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${HYP} --close-mosaic 15"
### (low) export TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${HYP} --optimizer AdamW --flat-cos-lr --close-mosaic 2 --save-period 1 --noplots"
## Parallel
#export CLIENT_NODES=1  # 目前每個Client只支援單節點多GPU運算，因為NCCL port 會衝突
export CLIENT_GPUS=2
export CLIENT_CPUS=8   # cpu = gpu x 4

#####################
## FL Server 端的參數
#####################
export SERVER_ALG="${SERVER_ALG:-fedawa}"   # 支持 fedavg, fedprox, fedavgm, fedopt, fedyoga, fedawa, fednova

# FedProx 演算法超參數（Server端命名
export SERVER_FEDPROX_MU=0.01  # FedProx 的 proximal term 係數
# FedAvgM 演算法超參數（Server端命名
export SERVER_FEDAVGM_LR=1.0  # FedAvgM 的 server learning rate
export SERVER_FEDAVGM_MOMENTUM=0.9 # FedAvgM 的 server momentum
# FedAWA 演算法超參數（Server端命名，基於原始論文）
export SERVER_FEDAWA_SERVER_EPOCHS=1  # 優化 T_weights 的輪數
export SERVER_FEDAWA_SERVER_OPTIMIZER="adam"  # 優化器: adam 或 sgd
export SERVER_FEDAWA_SERVER_LR=0.001  # Server端學習率 (adam:0.001, sgd:0.01，代碼會自動選擇)
export SERVER_FEDAWA_GAMMA=1.0  # 聚合權重縮放系數
export SERVER_FEDAWA_REG_DISTANCE="cos"  # 距離度量: cos (余弦) 或 euc (歐氏)
# 注意: T_weights 初始化策略
#   - 原論文: 第一輪基於客戶端數據量 [n1/N, n2/N, ..., nK/N]
#   - 本實作: 第一輪均勻初始化 [1/K, 1/K, ..., 1/K] (簡化版)
#   - 後續輪次會自動從 fedawa_state.pt 加載並持續優化
# FedYOGA 演算法超參數（Server端命名，自創算法）
export SERVER_FEDYOGA_HISTORY_WINDOW=3  # 降低歷史窗口避免過度累積
export SERVER_FEDYOGA_PCA_DIM=2  # 修正：應該遠小於客戶端數量，避免過擬合
export SERVER_FEDYOGA_SOFTMAX_TEMPERATURE=0.5
export SERVER_FEDYOGA_LOSSDROP_WEIGHT=1.0
export SERVER_FEDYOGA_GRADVAR_WEIGHT=1.0
export SERVER_FEDYOGA_PCA_SOLVER="full"
export SERVER_FEDYOGA_NORM_EPS=1e-8
# 修正：分階段裁剪策略
export SERVER_FEDYOGA_CLIP_THRESHOLD_PRE=1000.0  # PCA前粗略保護：避免極端值破壞PCA
export SERVER_FEDYOGA_CLIP_THRESHOLD_POST=10.0   # PCA後精細裁剪：保護降維後的特徵空間
export SERVER_FEDYOGA_ENABLE_BALANCE_WEIGHT=false  # IID 場景下關閉平衡權重
# FedOpt 演算法超參數（Server端命名）
export SERVER_FEDOPT_LR=0.001
export SERVER_FEDOPT_BETA1=0.9
export SERVER_FEDOPT_BETA2=0.999
export SERVER_FEDOPT_EPS=1e-8
# FedNova 演算法超參數（Server端命名）
export SERVER_FEDNOVA_MU=0.0
export SERVER_FEDNOVA_LR=1.0

# 目前 scaffold 尚未實作 client 端的 local control_variate 與 server control_variate
