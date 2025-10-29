#!/bin/bash

# ===================================================================================
# Orchestrator Script for Federated Learning Experiments (V3)
# ===================================================================================
# This script executes the entire FL pipeline automatically. It submits client jobs
# in parallel and uses Slurm dependencies to trigger the averaging step.
# Usage: ./src/orchestrate.sh <dataset> <clients> <rounds> [--val]
#
# Optional flags:
#    --val:    Include model validation step
# ===================================================================================

set -e
set -o pipefail

# --- 1. Argument Parsing ---
VALIDATION_ENABLED=false

# Parse optional flags
args=("$@")
filtered_args=()

for arg in "${args[@]}"; do
    if [[ "$arg" == "--val" ]]; then
        VALIDATION_ENABLED=true
    else
        filtered_args+=("$arg")
    fi
done

# Set filtered arguments back to positional parameters
set -- "${filtered_args[@]}"

# --- 2. Environment and Path Setup ---
source "${WROOT}/src/env.sh"


# --- 3. Configuration ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <DATASET_NAME> <CLIENT_NUM> <TOTAL_ROUNDS> [--val]"
    echo "ex : $0 kitti 4 2"
    echo "ex : $0 kitti 4 2 --val"
    exit 1
fi
DATASET_NAME=$1
CLIENT_NUM=$2
TOTAL_ROUNDS=$3
INITIAL_WEIGHTS="yolov9-c.pt"



# --- 4. Handle EXP_ID (from argument or auto-generate) ---
EXP_ID=$4
EXPERIMENTS_BASE_DIR="experiments"
if [ -z "$EXP_ID" ]; then
    mkdir -p ${EXPERIMENTS_BASE_DIR}
    RUN_COUNT=$(find "${EXPERIMENTS_BASE_DIR}" -maxdepth 1 -type d | wc -l)
    RUN_NUM=$((RUN_COUNT))
    TIMESTAMP=$(date +%Y%m%d%H%M)
    EXP_ID="${RUN_NUM}_${DATASET_NAME}_${SERVER_ALG}_${CLIENT_NUM}C_${TOTAL_ROUNDS}R_${TIMESTAMP}"
fi

EXP_DIR="${EXPERIMENTS_BASE_DIR}/${EXP_ID}"

# --- 4.1. Export Experiment Variables ---
# Export experiment identifiers for both modes

export WROOT="${WROOT}"
export SRC_DIR="${WROOT}/src"
export DATASET_NAME="${DATASET_NAME}"
export CLIENT_NUM="${CLIENT_NUM}"
export TOTAL_ROUNDS="${TOTAL_ROUNDS}"
export EXP_ID="${EXP_ID}"
export EXP_DIR="${EXP_DIR}"

# ===================================================================================
#
#  RUN THE FULL PIPELINE
#
# ===================================================================================

# --- 5. Create Directories ---
echo "######################################################################"
echo "##  Initializing Experiment Directories"
echo "##  Experiment ID: ${EXP_ID}"
echo "######################################################################"
mkdir -p "${EXP_DIR}/slurm_logs"
mkdir -p "${EXP_DIR}/client_outputs/${EXP_ID}"  # New structure: all rounds under EXP_ID
mkdir -p "${EXP_DIR}/aggregated_weights"
mkdir -p "${EXP_DIR}/fed_agg_logs"
cp "${WROOT}/src/env.sh" "${EXP_DIR}/env.sh"

# --- 6. Setup Logging ---
# Redirect all output to both console and log file
exec > >(tee -a "${EXP_DIR}/orchestrator.log")
exec 2> >(tee -a "${EXP_DIR}/orchestrator.log" >&2)

echo "##  STARTING PARALLEL FEDERATED LEARNING EXPERIMENT (AUTOMATED V3)"
echo "##  Experiment Directory: ${EXP_DIR}"
echo "##  Environment: ${EXP_DIR}/env.sh"

#echo -e "\n--- STEP 0: Preparing data for ${CLIENT_NUM} clients... ---"
#python3 "${SRC_DIR}/data_prepare.py" --dataset-name "${DATASET_NAME}" --num-clients "${CLIENT_NUM}"
#echo "--- Data preparation step complete. ---"

echo -e "\n--- STEP 1: Starting Federated Learning Rounds... ---"
for r in $(seq 1 ${TOTAL_ROUNDS}); do
    echo -e "\n==================[ ROUND ${r} / ${TOTAL_ROUNDS} ]=================="

    if [ "${r}" -eq 1 ]; then
        current_weights="${WROOT}/${INITIAL_WEIGHTS}"
    else
        prev_round=$((r - 1))
        current_weights="${WROOT}/${EXP_DIR}/aggregated_weights/w_s_r${prev_round}.pt"
    fi
    echo "Using weights for this round: ${current_weights}"

        # === Dynamic Hyperparameter Strategy: Switch to FedYOGA HYP from Round 2 ===
        if [ "${r}" -ge 2 ] && [ -n "${FedyogaHYP}" ] && [ -f "${FedyogaHYP}" ]; then
            echo "[STRATEGY] Round ${r}: Switching to FedYOGA hyperparameters (${FedyogaHYP})"
            CURRENT_HYP="${FedyogaHYP}"
        else
            echo "[STRATEGY] Round ${r}: Using standard hyperparameters (${HYP})"
            CURRENT_HYP="${HYP}"
        fi
    
        # Rebuild TRAIN_EXTRA_ARGS with the current hyperparameter file
        export TRAIN_EXTRA_ARGS="--epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --img ${IMG_SIZE} --workers ${WORKER} --hyp ${CURRENT_HYP} --close-mosaic 15"
        echo "[STRATEGY] Training arguments: ${TRAIN_EXTRA_ARGS}"

    # Submit all client jobs for the current round in parallel
    # The fl_client.sh script returns a list of submitted job IDs
    output=$("${SRC_DIR}/fl_client_train.sh" \
        "${WROOT}/${EXP_DIR}" \
        "${WROOT}" \
        "${r}" \
        "${CLIENT_NUM}" \
        "${DATASET_NAME}" \
        "${current_weights}")

    # Extract job IDs from the output of fl_client.sh
    # Assuming sbatch output is "Submitted batch job XXXXXX"
    client_job_ids=$(echo "${output}" | grep "Submitted batch job" | awk '{print $4}' | tr '\n' ' ')
    
    # Check if we have job IDs
    if [ -z "${client_job_ids}" ]; then
        echo "Error: Failed to submit client jobs or parse job IDs for round ${r}. Exiting." >&2
        exit 1
    fi

    dependency_list=$(echo ${client_job_ids} | sed 's/ /:/g')
    echo -e "\n--> All client jobs submitted. Dependency list: ${dependency_list}"
    echo ">> Submitting federated aggregation job, which will run after clients finish."

    # Submit the fed_agg job using the helper script, with dependency and wait flags
    "${SRC_DIR}/fl_server_fedagg.sh" \
        "${WROOT}/${EXP_DIR}" \
        "${WROOT}" \
        "${r}" \
        "${CLIENT_NUM}" \
        --algorithm ${SERVER_ALG} \
        --dependency "${dependency_list}" \
        --wait

    if [ $? -ne 0 ]; then
        echo "Error: Federated averaging failed for round ${r}. Stopping experiment."
        exit 1
    else
        echo "--> Federated averaging for Round ${r} complete."
    fi
done

# Run validation only if --val flag is provided
if [ "$VALIDATION_ENABLED" = true ]; then
    echo -e "\n--- STEP 3: Running Model Validation ---"
    echo ">> Validating all models (baseline + ${TOTAL_ROUNDS} rounds)..."

    # Run validation with error handling
    if python3 "${SRC_DIR}/validate_federated_model.py" \
        --experiment-dir "${WROOT}/${EXP_DIR}" \
        --data-config "data/${DATASET_NAME}.yaml" \
        ; then
        echo "--- Model validation complete. ---"
        VALIDATION_MSG="##  Validation results: ${EXP_DIR}/validation/"
    else
        echo "Warning: Model validation failed, but experiment completed successfully."
        VALIDATION_MSG="##  Validation: Failed (see logs above)"
    fi
else
    VALIDATION_MSG="##  Validation: Skipped (use --val to enable)"
fi

echo -e "\n######################################################################"
echo "##"
echo "##  AUTOMATED FEDERATED LEARNING EXPERIMENT COMPLETED"
echo "##  Final model: ${EXP_DIR}/aggregated_weights/w_s_r${TOTAL_ROUNDS}.pt"
echo "${VALIDATION_MSG}"
echo "##"
echo "######################################################################"
