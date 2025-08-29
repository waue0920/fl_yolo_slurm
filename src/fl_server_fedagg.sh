#!/bin/bash
#
# Shell Script to Submit Federated Aggregation Job for a Specific Round
#
# 依 fl_server_fedavg.sh 結構，支援多聚合演算法


set -e
set -o pipefail

# --- 1. Argument Parsing ---
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <EXP_DIR> <WROOT> <ROUND> <CLIENT_NUM> [--algorithm <name>] [--dependency <job_ids>] [--wait]"
    echo ""
    echo "Arguments:"
    echo "  EXP_DIR     : Full path to experiment directory"
    echo "  WROOT       : Full path to project root directory"
    echo "  ROUND       : Round number (1, 2, 3, ...)"
    echo "  CLIENT_NUM  : Number of clients expected"
    echo ""
    echo "Optional arguments:"
    echo "  --algorithm <name>      : Aggregation algorithm (fedavg, fedprox, scaffold)"
    echo "  --dependency <job_ids>  : Comma-separated list of job IDs to wait for"
    echo "  --wait                  : Wait for job completion (for automated mode)"
    echo ""
    echo "Examples:"
    echo "  Manual mode:  $0 /path/experiments/exp_id /path/project_root 1 4 --algorithm fedprox"
    echo "  Auto mode:    $0 /path/experiments/exp_id /path/project_root 1 4 --algorithm scaffold --dependency 123:124:125 --wait"
    exit 1
fi

EXP_DIR=$1
WROOT=$2
ROUND=$3
CLIENT_NUM=$4
shift 4

# Extract EXP_ID from EXP_DIR for path construction
EXP_ID=$(basename "${EXP_DIR}")

# --- 2. Parse Optional Arguments ---
ALGORITHM="${SERVER_ALG:-fedavg}"
DEPENDENCY=""
WAIT_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --dependency)
            DEPENDENCY="$2"
            shift 2
            ;;
        --wait)
            WAIT_FLAG="--wait"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --- 3. Generate Required Paths ---
INPUT_DIR="${EXP_DIR}/client_outputs/${EXP_ID}"
OUTPUT_FILE="${EXP_DIR}/aggregated_weights/w_s_r${ROUND}.pt"
LOG_OUT="${EXP_DIR}/fed_agg_logs/round_${ROUND}.out"
LOG_ERR="${EXP_DIR}/fed_agg_logs/round_${ROUND}.err"

echo "======================================================================"
echo "## Submitting federated aggregation job for Round ${ROUND}"
echo "## Experiment Directory: ${EXP_DIR}"
echo "## Expected Clients: ${CLIENT_NUM}"
echo "## Input Directory: ${INPUT_DIR}"
echo "## Output Model: ${OUTPUT_FILE}"
echo "## Algorithm: ${ALGORITHM}"
if [ -n "${DEPENDENCY}" ]; then
    echo "## Dependency Jobs: ${DEPENDENCY}"
fi

# --- 4. Build sbatch command with optional arguments (Bash array style) ---
SBATCH_CMD=(sbatch --export=ALL --job-name="fed_agg_r${ROUND}" --output="${LOG_OUT}" \
    --error="${LOG_ERR}" --chdir="${WROOT}")

if [ -n "${DEPENDENCY}" ]; then
    SBATCH_CMD+=(--dependency=afterok:${DEPENDENCY})
fi
if [ -n "${WAIT_FLAG}" ]; then
    SBATCH_CMD+=(${WAIT_FLAG})
fi

SBATCH_CMD+=("${WROOT}/src/server_fedagg.sb" \
    --input-dir "${INPUT_DIR}" \
    --output-file "${OUTPUT_FILE}" \
    --expected-clients "${CLIENT_NUM}" \
    --round "${ROUND}" \
    --algorithm "${ALGORITHM}")

# --- 5. Execute the sbatch command ---
echo "${SBATCH_CMD[@]}"
echo "======================================================================"
"${SBATCH_CMD[@]}"

echo "----------------------------------------------------------------------"
echo "## Federated aggregation job for Round ${ROUND} has been submitted."
echo "## Monitor job status with: squeue -u \${USER}"
echo "## Check logs at: ${LOG_OUT}"
echo "======================================================================"
