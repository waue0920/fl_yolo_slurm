# 集中管理所有實驗參數
# 可在此修改所有訓練與資源參數

# 主程式相關設定
export SINGULARITY_IMG="${WROOT}/yolo9t2_ngc2306_20241226.sif"
export EXPERIMENTS_BASE_DIR="${WROOT}/experiments"


# Slurm 資源參數
export SLURM_PARTITION=gp2d
export SLURM_ACCOUNT="GOV113038"

# fl client 端的 slurm 參數
export CLIENT_NODES=1  # 目前只支援單節點，因為port 會衝突
export CLIENT_GPUS=2
export CLIENT_CPUS=8

# fl client 端的 訓練參數
export EPOCHS=100
export BATCH_SIZE=16
export IMG_SIZE=640



