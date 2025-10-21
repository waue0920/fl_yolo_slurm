#!/bin/bash

# ===================================================================================
# Replay while Orchestrator Script Failed
# ===================================================================================
# Usage: ./src/replay.sh <exp_dir_path>
# ex : ./src/replay.sh ./experiments/18_coco_4C_6R_202510010849
# ===================================================================================

# --- 1. Argument Parsing ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <EXPERIMENT_DIR>"
    exit 1
fi

EXP_DIR="$1"
EXP_ID=$(basename "$EXP_DIR")
EXPERIMENTS_BASE_DIR=$(dirname "$EXP_DIR")

# --- 2. Environment and Path Setup ---
source "${EXP_DIR}/env.sh"
## 皆以定義於 ${EXP_DIR}/env.sh
# DATASET_NAME=$1
# CLIENT_NUM=$2
# TOTAL_ROUNDS=$3


# --- 3. Export Experiment Variables ---
# Export experiment identifiers

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


# --- 4. Setup Logging ---
# Redirect all output to both console and log file
exec > >(tee -a "${EXP_DIR}/orchestrator.log")
exec 2> >(tee -a "${EXP_DIR}/orchestrator.log" >&2)
echo "##  STARTING NEW PARALLEL FEDERATED LEARNING EXPERIMENT (AUTOMATED V3)"
echo "##  Experiment Directory: ${EXP_DIR}"
echo "##  Environment: ${EXP_DIR}/env.sh"


# --- 計算已完成的最大 round ---
max_completed_round=0
for ((r=1; r<=TOTAL_ROUNDS; r++)); do
    if [ -f "${EXP_DIR}/aggregated_weights/w_s_r${r}.pt" ]; then
        max_completed_round=$r
    else
        break
    fi
done

start_round=$((max_completed_round + 1))
if [ $start_round -gt $TOTAL_ROUNDS ]; then
    echo "All rounds already completed. Nothing to replay."
    exit 0
fi

# --- 5. Continue Run ---
echo -e "\n--- STEP 1: Continue Federated Learning Rounds from Round ${start_round}... ---"
for ((r=$start_round; r<=TOTAL_ROUNDS; r++)); do
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

    # Submit all client jobs for the current round in parallel
    # The fl_client.sh script returns a list of submitted job IDs
    echo "[EXECUTING] Client training for round ${r}:"
    set -x
    output=$("${SRC_DIR}/fl_client_train.sh" \
        "${WROOT}/${EXP_DIR}" \
        "${WROOT}" \
        "${r}" \
        "${CLIENT_NUM}" \
        "${DATASET_NAME}" \
        "${current_weights}")
    set +x

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
    echo "[EXECUTING] Federated aggregation for round ${r}:"
    set -x
    "${SRC_DIR}/fl_server_fedagg.sh" \
        "${WROOT}/${EXP_DIR}" \
        "${WROOT}" \
        "${r}" \
        "${CLIENT_NUM}" \
        --algorithm ${SERVER_ALG} \
        --dependency "${dependency_list}" \
        --wait
    set +x

    if [ $? -ne 0 ]; then
        echo "Error: Federated averaging failed for round ${r}. Stopping experiment."
        exit 1
    else
        echo "--> Federated Aggregate for Round ${r} complete."
    fi
done


echo -e "\n######################################################################"
echo "##"
echo "##  AUTOMATED FEDERATED LEARNING EXPERIMENT COMPLETED"
echo "##  Final model: ${EXP_DIR}/aggregated_weights/w_s_r${TOTAL_ROUNDS}.pt"
echo "##"
echo "######################################################################"
