#!/bin/bash

# ===================================================================================
# Replay Standalone Script - Resume Failed Standalone Orchestrator Execution
# ===================================================================================
# This script resumes a failed standalone orchestrator run from the last completed round.
# It executes sequentially in conda environment without SLURM.
# Usage: ./src/replay_standalone.sh <exp_dir_path> [--dry-run]
# ex : ./src/replay_standalone.sh ./experiments/48_kittiO_fedawa_10C_12R_202511040953
# ex : ./src/replay_standalone.sh ./experiments/48_kittiO_fedawa_10C_12R_202511040953 --dry-run
# ===================================================================================

set -e
set -o pipefail

# --- 1. Argument Parsing ---
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <EXPERIMENT_DIR> [--dry-run]"
    echo "Example: $0 ./experiments/48_kittiO_fedawa_10C_12R_202511040953"
    exit 1
fi

EXP_DIR="$1"
DRY_RUN=false

# Make EXP_DIR absolute, similar to standalone_orchestrate.sh
EXP_DIR="$(cd "${EXP_DIR}" && pwd)"


shift
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

# Print dry-run banner if enabled
if [ "$DRY_RUN" = true ]; then
    echo "=========================================="
    echo "        DRY RUN MODE ENABLED"
    echo "=========================================="
    echo "No actual training or aggregation will be performed."
    echo "Commands will be displayed for review only."
    echo "=========================================="
    echo ""
fi

# Validate experiment directory exists
if [ ! -d "${EXP_DIR}" ]; then
    echo "Error: Experiment directory not found: ${EXP_DIR}"
    exit 1
fi

EXP_ID=$(basename "$EXP_DIR")
EXPERIMENTS_BASE_DIR=$(dirname "$EXP_DIR")

# --- 2. Environment and Path Setup ---
# Find config file matching *_env.sh pattern in experiment directory
CONF_FILE=$(find "${EXP_DIR}" -maxdepth 1 -name "*env.sh" -type f | head -n 1)

if [ -z "${CONF_FILE}" ]; then
    echo "Error: No configuration file matching '*env.sh' found in ${EXP_DIR}"
    exit 1
fi

echo "Found configuration file: ${CONF_FILE}"
source "${CONF_FILE}"

# --- 3. Export Experiment Variables ---
export WROOT="${WROOT}"
export SRC_DIR="${WROOT}/src"
export DATASET_NAME="${DATASET_NAME}"
export CLIENT_NUM="${CLIENT_NUM}"
export TOTAL_ROUNDS="${TOTAL_ROUNDS}"
export EXP_ID="${EXP_ID}"
export EXP_DIR="${EXP_DIR}"

# NCCL Setup (shared across all clients since they run sequentially)
get_free_port() {
    while :; do
        PORT=$(( ( RANDOM % 50000 )  + 10000 ))
        if ! lsof -i:"$PORT" >/dev/null 2>&1; then
            echo "$PORT"
            return
        fi
    done
}
MASTER_PORT=$(get_free_port)
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost

# --- GPU Requirement Check ---
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ $AVAILABLE_GPUS -lt 1 ]; then
    echo "=========================================="
    echo "ERROR: GPU Requirement Not Met"
    echo "=========================================="
    exit 1
fi

cd "${WROOT}"
echo "GPU Check: ${AVAILABLE_GPUS} GPU(s) detected - OK"

# Respect MODEL_CFG set by configuration; fallback to YOLOv9 default
MODEL_CFG="${MODEL_CFG:-${WROOT}/yolov9/models/detect/yolov9-c.yaml}"

# ===================================================================================
#
#  RUN THE FULL PIPELINE (RESUME FROM LAST COMPLETED ROUND)
#
# ===================================================================================

# --- 4. Setup Logging ---
exec > >(tee -a "${EXP_DIR}/orchestrator.log")
exec 2>&1

echo "##  RESUMING STANDALONE FEDERATED LEARNING EXPERIMENT"
echo "##  Experiment Directory: ${EXP_DIR}"
echo "##  Configuration: ${CONF_FILE}"
echo "##  DRY_RUN: ${DRY_RUN}"

# --- 5. Calculate Last Completed Round ---
echo -e "\n--- Detecting last completed round... ---"
max_completed_round=0
for ((r=1; r<=TOTAL_ROUNDS; r++)); do
    if [ -f "${EXP_DIR}/aggregated_weights/w_s_r${r}.pt" ]; then
        max_completed_round=$r
        echo ">> Found completed round: ${r}"
    else
        break
    fi
done

start_round=$((max_completed_round + 1))
if [ $start_round -gt $TOTAL_ROUNDS ]; then
    echo "All rounds already completed. Nothing to replay."
    exit 0
fi

echo ">> Last completed round: ${max_completed_round}"
echo ">> Resuming from round: ${start_round}"

# --- 6. Resume Federated Learning Rounds ---
echo -e "\n--- STEP 1: Resuming Federated Learning Rounds from Round ${start_round}... ---"
for r in $(seq ${start_round} ${TOTAL_ROUNDS}); do
    echo -e "\n==================[ ROUND ${r} / ${TOTAL_ROUNDS} ]=================="

    if [ "${r}" -eq 1 ]; then
        if [ -n "${INITIAL_WEIGHTS}" ] && [ ! -f "${WROOT}/${INITIAL_WEIGHTS}" ]; then
            echo "Error: INITIAL_WEIGHTS is set to '${INITIAL_WEIGHTS}', but the file was not found at '${WROOT}/${INITIAL_WEIGHTS}'." >&2
            exit 1
        fi
        current_weights="${WROOT}/${INITIAL_WEIGHTS}"
    else
        prev_round=$((r - 1))
        current_weights="${EXP_DIR}/aggregated_weights/w_s_r${prev_round}.pt"
    fi

    echo "Using weights for this round: ${current_weights}"

    # === Dynamic Hyperparameter Strategy ===
    # Default: FedyogaHYP | Use HYP only if: FL_HYP_THRESHOLD=0 OR r<=FL_HYP_THRESHOLD
    if [ "${FL_HYP_THRESHOLD}" = "" ]; then
        CURRENT_HYP="${FedyogaHYP}"
    elif [ "${r}" -le "${FL_HYP_THRESHOLD}" ] || [ "${FL_HYP_THRESHOLD}" = "0" ]; then
        CURRENT_HYP="${HYP}"
    else
        CURRENT_HYP="${FedyogaHYP}"
    fi
    
    TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${CURRENT_HYP} --close-mosaic 15"
    echo "[STRATEGY] Round ${r}: Using $(basename ${CURRENT_HYP}) | Args: ${TRAIN_EXTRA_ARGS}"

    # Execute all client jobs for the current round SEQUENTIALLY
    echo "[EXECUTING] Sequential client training for round ${r}:"
    for c in $(seq 1 ${CLIENT_NUM}); do
        echo -e "\n--- Training Client ${c}/${CLIENT_NUM} ---"

        # Per-client configuration
        CLIENT_DATA_YAML="${WROOT}/federated_data/${DATASET_NAME}_${CLIENT_NUM}/c${c}.yaml"
        OUTPUT_NAME="r${r}_c${c}"
        WANDB_RUN_NAME="r${r}_c${c}"
        PROJECT_OUT="${EXP_DIR}/client_outputs/${EXP_ID}"

        # GPU configuration
        NPROC_PER_NODE=${AVAILABLE_GPUS}
        DEVICE_LIST=$(seq -s, 0 $(($AVAILABLE_GPUS-1)) | paste -sd, -)

        echo ">> Client ${c} Configuration:"
        echo ">>   Data YAML:       ${CLIENT_DATA_YAML}"
        echo ">>   Input Weights:   ${current_weights}"
        echo ">>   Project:         ${PROJECT_OUT}"
        echo ">>   Run Name:        ${WANDB_RUN_NAME}"
        echo ">>   GPUs:            ${NPROC_PER_NODE}"
        echo ">>   Device List:     ${DEVICE_LIST}"
        echo ">>   Master Port:     ${MASTER_PORT}"

        # Check if files exist
        if [ ! -f "${CLIENT_DATA_YAML}" ]; then
            echo "Error: Data YAML file not found at ${CLIENT_DATA_YAML}"
            exit 1
        fi

        # Execute YOLOv9 Training in Conda Environment
        cd "${WROOT}"/yolov9

        # Build torchrun command
        TORCHRUN_CMD=(
            torchrun --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}"
            --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}"
            "./train_dual.py"
            --weights "${current_weights}"
            --data "${CLIENT_DATA_YAML}"
            --cfg "${MODEL_CFG}"
            --project "${PROJECT_OUT}"
            --name "${WANDB_RUN_NAME}"
            --device "${DEVICE_LIST}"
            --exist-ok
        )

        # Add extra training arguments
        if [ -n "${TRAIN_EXTRA_ARGS}" ]; then
            TORCHRUN_CMD+=( ${TRAIN_EXTRA_ARGS} )
        fi

        echo "!! [replay_standalone] Direct FL training execution @ Conda Environment !!"
        
        if [ "$DRY_RUN" = true ]; then
            echo ""
            echo "[DRY-RUN] Would execute training command:"
            echo "----------------------------------------"
            printf '%q ' "${TORCHRUN_CMD[@]}"
            echo ""
            echo "----------------------------------------"
            echo "[DRY-RUN] Skipping actual execution"
            TRAINING_EXIT_CODE=0
        else
            set -x
            "${TORCHRUN_CMD[@]}"
            TRAINING_EXIT_CODE=$?
            set +x
        fi

        if [ ${TRAINING_EXIT_CODE} -ne 0 ]; then
            echo "Error: Client ${c} training failed for round ${r} with exit code ${TRAINING_EXIT_CODE}. Stopping experiment."
            exit 1
        else
            echo "--> Client ${c} training for Round ${r} complete."
        fi
    done

    echo -e "\n--> All client training for round ${r} completed successfully."

    # Execute federated aggregation
    echo "[EXECUTING] Federated aggregation for round ${r}:"

    INPUT_DIR="${EXP_DIR}/client_outputs/${EXP_ID}"
    OUTPUT_FILE="${EXP_DIR}/aggregated_weights/w_s_r${r}.pt"

    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "[DRY-RUN] Would execute aggregation command:"
        echo "----------------------------------------"
        echo "python3 ${SRC_DIR}/server_fedagg.py \\"
        echo "    --input-dir \"${INPUT_DIR}\" \\"
        echo "    --output-file \"${OUTPUT_FILE}\" \\"
        echo "    --expected-clients \"${CLIENT_NUM}\" \\"
        echo "    --round \"${r}\" \\"
        echo "    --algorithm \"${SERVER_ALG}\""
        echo "----------------------------------------"
        echo "[DRY-RUN] Skipping actual execution"
        AGG_EXIT_CODE=0
    else
        set -x
        python3 "${SRC_DIR}/server_fedagg.py" \
            --input-dir "${INPUT_DIR}" \
            --output-file "${OUTPUT_FILE}" \
            --expected-clients "${CLIENT_NUM}" \
            --round "${r}" \
            --algorithm "${SERVER_ALG}"
        AGG_EXIT_CODE=$?
        set +x
    fi

    if [ $AGG_EXIT_CODE -ne 0 ]; then
        echo "Error: Federated averaging failed for round ${r}. Stopping experiment."
        exit 1
    else
        echo "--> Federated Aggregate for Round ${r} complete."
    fi
done

echo -e "\n######################################################################"
echo "##"
if [ "$DRY_RUN" = true ]; then
    echo "##  DRY-RUN REPLAY COMPLETED (No actual execution performed)"
else
    echo "##  STANDALONE FEDERATED LEARNING REPLAY COMPLETED"
fi
echo "##  Experiment ID: ${EXP_ID}"
if [ "$DRY_RUN" = true ]; then
    echo "##  Final model would be: ${EXP_DIR}/aggregated_weights/w_s_r${TOTAL_ROUNDS}.pt"
else
    echo "##  Final model: ${EXP_DIR}/aggregated_weights/w_s_r${TOTAL_ROUNDS}.pt"
fi
echo "##"
echo "######################################################################"
