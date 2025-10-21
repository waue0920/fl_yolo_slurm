#!/bin/bash

# ===================================================================================
# Replay Standalone Script - Resume Failed Standalone Orchestrator Execution
# ===================================================================================
# This script resumes a failed standalone orchestrator run from the last completed round.
# It executes sequentially in conda environment without SLURM.
# Usage: ./src/replay_standalone.sh <exp_dir_path> [--dry-run]
# ex : ./src/replay_standalone.sh ./experiments/18_coco_4C_6R_202510010849
# ex : ./src/replay_standalone.sh ./experiments/18_coco_4C_6R_202510010849 --dry-run
# ===================================================================================

set -e
set -o pipefail

# --- 1. Argument Parsing ---
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <EXPERIMENT_DIR> [--dry-run]"
    exit 1
fi

EXP_DIR="$1"
DRY_RUN=false

# Parse optional flags
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

EXP_ID=$(basename "$EXP_DIR")
EXPERIMENTS_BASE_DIR=$(dirname "$EXP_DIR")

# --- 2. Environment and Path Setup ---
if [ ! -f "${EXP_DIR}/env.sh" ]; then
    echo "Error: Environment file not found at ${EXP_DIR}/env.sh"
    exit 1
fi

source "${EXP_DIR}/env.sh"
## Variables defined in ${EXP_DIR}/env.sh:
# WROOT
# DATASET_NAME
# CLIENT_NUM
# TOTAL_ROUNDS
# SERVER_ALG
# INITIAL_WEIGHTS
# BATCH_SIZE
# EPOCHS
# TRAIN_EXTRA_ARGS

# --- 3. Export Experiment Variables ---
export WROOT="${WROOT}"
export SRC_DIR="${WROOT}/src"
export DATASET_NAME="${DATASET_NAME}"
export CLIENT_NUM="${CLIENT_NUM}"
export TOTAL_ROUNDS="${TOTAL_ROUNDS}"
export EXP_ID="${EXP_ID}"
export EXP_DIR="${EXP_DIR}"

# --- GPU Requirement Check ---
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ $AVAILABLE_GPUS -lt 1 ]; then
    echo "=========================================="
    echo "ERROR: GPU Requirement Not Met"
    echo "=========================================="
    exit 1
fi

echo "GPU Check: ${AVAILABLE_GPUS} GPU(s) detected - OK"

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
echo "##  Environment: ${EXP_DIR}/env.sh"
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
        # If INITIAL_WEIGHTS is set, verify it exists. If empty, it's a valid scratch training.
        if [ -n "${INITIAL_WEIGHTS}" ] && [ ! -f "${INITIAL_WEIGHTS}" ]; then
            echo "Error: INITIAL_WEIGHTS is set to '${INITIAL_WEIGHTS}' in env.sh, but the file was not found at '${INITIAL_WEIGHTS}'." >&2
            exit 1
        fi
        current_weights="${INITIAL_WEIGHTS}"
    else
        prev_round=$((r - 1))
        current_weights="${WROOT}/${EXP_DIR}/aggregated_weights/w_s_r${prev_round}.pt"
    fi

    echo "Using weights for this round: ${current_weights}"

    # Execute all client jobs for the current round SEQUENTIALLY
    echo "[EXECUTING] Sequential client training for round ${r}:"
    for c in $(seq 1 ${CLIENT_NUM}); do
        echo -e "\n--- Training Client ${c}/${CLIENT_NUM} ---"

        # Per-client configuration
        CLIENT_DATA_YAML="${WROOT}/federated_data/${DATASET_NAME}_${CLIENT_NUM}/c${c}.yaml"
        OUTPUT_NAME="r${r}_c${c}"
        WANDB_RUN_NAME="r${r}_c${c}"
        PROJECT_OUT="${WROOT}/${EXP_DIR}/client_outputs/${EXP_ID}"

        # === Integrated Client Training Logic (Conda Environment) ===
        if [ -z "${WROOT}" ]; then
            echo "Error: WROOT environment variable is not set"
            exit 1
        fi

        MODEL_CFG="${WROOT}/gyolo/models/caption/gyolo.yaml"

        # Dynamic port allocation for each client
        MASTER_PORT=59527
        NNODES=1
        NODE_RANK=0
        MASTER_ADDR=localhost

        # GPU configuration - use detected GPU count from system check
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

        # Check if files exist before proceeding
        if [ ! -f "${CLIENT_DATA_YAML}" ]; then
            echo "Error: Data YAML file not found at ${CLIENT_DATA_YAML}"
            exit 1
        fi

        # Execute YOLOv9 Training in Conda Environment
        cd "${WROOT}/gyolo"

        # Build torchrun command for conda environment
        TORCHRUN_CMD=(
            torchrun --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}"
            --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}"
            "./caption/train.py"
            --weights "${current_weights}"
            --data "${CLIENT_DATA_YAML}"
            --cfg "${MODEL_CFG}"
            --project "${PROJECT_OUT}"
            --name "${WANDB_RUN_NAME}"
            --device "${DEVICE_LIST}"
            --exist-ok
        )

        # Add extra training arguments if provided
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

    # Execute federated aggregation directly (no sbatch)
    echo "[EXECUTING] Federated aggregation for round ${r}:"

    # Generate required paths
    INPUT_DIR="${WROOT}/${EXP_DIR}/client_outputs/${EXP_ID}"
    OUTPUT_FILE="${WROOT}/${EXP_DIR}/aggregated_weights/w_s_r${r}.pt"

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
        echo "--> Federated averaging for Round ${r} complete."
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
