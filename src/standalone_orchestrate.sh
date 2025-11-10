#!/bin/bash

# ===================================================================================
# Standalone Orchestrator Script for YOLOv9 Federated Learning
# ===================================================================================
# This script is completely independent of SLURM and integrates all launch logic.
# It executes the entire FL pipeline sequentially in conda environment.
# Primary configuration is read from a config shell file (env-style exports).
# Usage: ./src/standalone_orchestrate.sh --conf /path/to/your_env.sh [--dry-run]
#
# Required flags:
#    --conf <file> : Path to an env-style bash file exporting variables like
#                    WROOT, DATASET_NAME, CLIENT_NUM, TOTAL_ROUNDS, etc.
# Optional flags:
#    --dry-run     : Show what would be executed without actually running
# ===================================================================================

set -e
set -o pipefail

# --- 1. Parse Flags ---

VALIDATION_ENABLED=true
DRY_RUN=false
CONF_FILE=""

# Simple argument parsing. Supports:
#   --conf <file>
#   --dry-run
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conf)
            CONF_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            # ignore unknown args for forward compatibility
            shift
            ;;
    esac
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

# --- 2. Auto-detect WROOT and Load Environment Configuration ---

# Detect default WROOT based on script location (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTED_WROOT="$(dirname "${SCRIPT_DIR}")"

# Load configuration from --conf if provided; otherwise fallback to default env.sh
CONF_PROVIDED=false
if [ -n "${CONF_FILE}" ]; then
    if [ ! -f "${CONF_FILE}" ]; then
        echo "Error: --conf specified but file not found: ${CONF_FILE}"
        exit 1
    fi
    echo "Using configuration file: ${CONF_FILE}"
    # Do not override WROOT before sourcing; let the config define it.
    # shellcheck disable=SC1090
    source "${CONF_FILE}"
    CONF_PROVIDED=true
    # If config didn't define WROOT, fallback to detected one
    if [ -z "${WROOT}" ]; then
        export WROOT="${DETECTED_WROOT}"
        echo "WROOT not defined in config; falling back to detected: ${WROOT}"
    fi
else
    # No --conf provided; use default project env
    export WROOT="${DETECTED_WROOT}"
    echo "Auto-detected WROOT: ${WROOT}"
    echo "Import ${WROOT}/src/env.sh"
    # shellcheck disable=SC1090
    source "${WROOT}/src/env.sh"
    CONF_FILE="${WROOT}/src/env.sh"
fi


# NCCL Setup
get_free_port() {
    while :; do
        PORT=$(( ( RANDOM % 50000 )  + 10000 ))
        # 檢查 port 是否已被使用
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



# Validate required parameters from configuration
if [ -z "${DATASET_NAME}" ] || [ -z "${CLIENT_NUM}" ] || [ -z "${TOTAL_ROUNDS}" ]; then
    echo "Error: Required parameters not set in configuration file (${CONF_FILE})"
    echo "Please ensure DATASET_NAME, CLIENT_NUM, and TOTAL_ROUNDS are exported in ${CONF_FILE}"
    exit 1
fi

# --- GPU Requirement Check ---
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ $AVAILABLE_GPUS -lt 1 ]; then
    echo "=========================================="
    echo "ERROR: GPU Requirement Not Met"
    echo "=========================================="
    exit 1
fi

echo "GPU Check: ${AVAILABLE_GPUS} GPU(s) detected - OK"

# Change to project root directory
cd "${WROOT}"

# --- 3. Generate Experiment ID ---

TIMESTAMP=$(date +%Y%m%d%H%M)

if [ -d "${EXPERIMENTS_BASE_DIR}" ]; then
    RUN_COUNT=$(find "${EXPERIMENTS_BASE_DIR}" -maxdepth 1 -type d 2>/dev/null | wc -l)
    RUN_NUM=$((RUN_COUNT))
else
    RUN_NUM=1
fi

export EXP_ID="${RUN_NUM}_${DATASET_NAME}_${SERVER_ALG}_${CLIENT_NUM}C_${TOTAL_ROUNDS}R_${TIMESTAMP}"

# Record environment configuration
echo "========== Environment Configuration =========="
echo "CONFIG_FILE:     ${CONF_FILE}"
echo "WROOT:           ${WROOT}"
echo "DATASET_NAME:    ${DATASET_NAME}"
echo "CLIENT_NUM:      ${CLIENT_NUM}"
echo "TOTAL_ROUNDS:    ${TOTAL_ROUNDS}"
echo "EXP_ID:          ${EXP_ID}"
echo "RUN_NUM:         ${RUN_NUM}"
echo "SERVER_ALG:      ${SERVER_ALG}"
echo "DETECTED_GPUS:   ${AVAILABLE_GPUS}"
echo "BATCH_SIZE:      ${BATCH_SIZE}"
echo "EPOCHS:          ${EPOCHS}"
echo "VALIDATION:      ${VALIDATION_ENABLED}"
echo "DRY_RUN:         ${DRY_RUN}"
echo "=================================================="

# --- 4. Setup Experiment Variables ---

export EXP_DIR="${EXPERIMENTS_BASE_DIR}/${EXP_ID}"
export SRC_DIR="${WROOT}/src"

# Respect MODEL_CFG set by configuration; fallback to YOLOv9 default
MODEL_CFG="${MODEL_CFG:-${WROOT}/yolov9/models/detect/yolov9-c.yaml}"

# ===================================================================================
#
#  RUN THE FULL PIPELINE (SEQUENTIAL EXECUTION)
#
# ===================================================================================



# --- 5. Create Directories ---
echo "######################################################################"
echo "##  Initializing Experiment Directories"
echo "##  Experiment ID: ${EXP_ID}"
echo "######################################################################"
mkdir -p "${EXP_DIR}/slurm_logs"
mkdir -p "${EXP_DIR}/client_outputs/${EXP_ID}"
mkdir -p "${EXP_DIR}/aggregated_weights"
mkdir -p "${EXP_DIR}/fed_agg_logs"
# Save the configuration used for this run into the experiment folder
if [ "${CONF_PROVIDED}" = true ]; then
    # Copy the provided config file into the experiment folder (keep original name)
    cp "${CONF_FILE}" "${EXP_DIR}/"
else
    # No config provided: copy default and rename to def_env.sh for clarity
    cp "${WROOT}/src/env.sh" "${EXP_DIR}/def_env.sh"
fi

# --- 6. Setup Logging ---
exec > >(tee -a "${EXP_DIR}/orchestrator.log")
exec 2>&1

echo "##  STARTING STANDALONE FEDERATED LEARNING EXPERIMENT "
echo "##  Experiment Directory: ${EXP_DIR}"
echo "##  Environment: ${EXP_DIR}/env.sh"

echo -e "\n--- STEP 1: Starting Federated Learning Rounds... ---"
for r in $(seq 1 ${TOTAL_ROUNDS}); do
    echo -e "\n==================[ ROUND ${r} / ${TOTAL_ROUNDS} ]=================="

    if [ "${r}" -eq 1 ]; then
        if [ -n "${INITIAL_WEIGHTS}" ] && [ ! -f "${WROOT}/${INITIAL_WEIGHTS}" ]; then
            echo "Error: INITIAL_WEIGHTS is set to '${INITIAL_WEIGHTS}' in env.sh, but the file was not found at '${WROOT}/${INITIAL_WEIGHTS}'." >&2
            exit 1
        fi
        current_weights="${WROOT}/${INITIAL_WEIGHTS}"
    else
        prev_round=$((r - 1))
        current_weights="${EXP_DIR}/aggregated_weights/w_s_r${prev_round}.pt"
    fi

    echo "Using weights for this round: ${current_weights}"

    # === Dynamic Hyperparameter Strategy: Switch based on FL_HYP_THRESHOLD ===
    # FL_HYP_THRESHOLD = 0 or unset: disable switching, always use HYP
    # FL_HYP_THRESHOLD > 0: must be <= TOTAL_ROUNDS; switch to FedyogaHYP when round >= threshold
    if [ -z "${FL_HYP_THRESHOLD}" ] || [ "${FL_HYP_THRESHOLD}" -eq 0 ]; then
        # Disabled: always use standard HYP
        echo "[STRATEGY] Round ${r}: Hyperparameter switching disabled (FL_HYP_THRESHOLD=${FL_HYP_THRESHOLD:-unset})"
        CURRENT_HYP="${HYP}"
    else
        # Validate threshold is reasonable
        if [ "${FL_HYP_THRESHOLD}" -gt "${TOTAL_ROUNDS}" ]; then
            echo "Error: FL_HYP_THRESHOLD (${FL_HYP_THRESHOLD}) must be <= TOTAL_ROUNDS (${TOTAL_ROUNDS})." >&2
            exit 1
        fi
        
        # Apply switching logic
        if [ "${r}" -ge "${FL_HYP_THRESHOLD}" ] && [ -n "${FedyogaHYP}" ] && [ -f "${FedyogaHYP}" ]; then
            echo "[STRATEGY] Round ${r}: Switching to FedYOGA hyperparameters (${FedyogaHYP})"
            CURRENT_HYP="${FedyogaHYP}"
        else
            echo "[STRATEGY] Round ${r}: Using standard hyperparameters (${HYP})"
            CURRENT_HYP="${HYP}"
        fi
    fi
    
    # Rebuild TRAIN_EXTRA_ARGS with the current hyperparameter file
    TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${CURRENT_HYP} --close-mosaic 15"
    echo "[STRATEGY] Training arguments: ${TRAIN_EXTRA_ARGS}"

    # Execute all client jobs for the current round SEQUENTIALLY
    echo "[EXECUTING] Sequential client training for round ${r}:"
    for c in $(seq 1 ${CLIENT_NUM}); do
        echo -e "\n--- Training Client ${c}/${CLIENT_NUM} ---"

        # Per-client configuration
        CLIENT_DATA_YAML="${WROOT}/federated_data/${DATASET_NAME}_${CLIENT_NUM}/c${c}.yaml"
        OUTPUT_NAME="r${r}_c${c}"
        WANDB_RUN_NAME="r${r}_c${c}"
        PROJECT_OUT="${EXP_DIR}/client_outputs/${EXP_ID}"

    # === Integrated Client Training Logic (Conda Environment) ===



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
        cd "${WROOT}/yolov9"

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

        echo "!! [standalone_orchestrate] Direct FL training execution @ Conda Environment !!"
        
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
echo "##  STANDALONE FEDERATED LEARNING EXPERIMENT COMPLETED"
echo "##  Final model: ${EXP_DIR}/aggregated_weights/w_s_r${TOTAL_ROUNDS}.pt"
echo "##"
echo "######################################################################"

# Optional validation step
if [ "$VALIDATION_ENABLED" = true ]; then
    echo -e "\n--- STEP 2: Model Validation ---"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would execute validation using validate_models.sh"
    else
        bash "${SRC_DIR}/validate_models.sh" "${EXP_DIR}"
    fi
fi
