#!/bin/bash

set -euo pipefail

# validate_models.sh
# Simplified validator: validate.sh [MODEL_INPUT_DIR]
# Given an experiment directory, finds aggregated weights (w_s_r*.pt) in the experiment's
# directory (e.g., fed_agg_logs or aggregated_weights) and validates each round sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_VALIDATOR="${SCRIPT_DIR}/validate_federated_model.py"

show_help() {
    cat <<EOF
Usage: $0 <MODEL_INPUT_DIR>

Example:
  $0 ~/fl_yolo_slurm/archive/kittiA010_fedavg_4C_30R_5S_202510202356

The script will search for weight files named "w_s_r*.pt" in the experiment's
aggregated_weights or fed_agg_logs folder and validate each round sequentially.
EOF
}

if [ $# -ne 1 ]; then
    show_help
    exit 1
fi

EXP_DIR="$1"
DEVICE="0"
BASELINE="yolov9-c.pt"

# If experiment env.sh exists, source it to get DATASET_NAME, etc.
if [ -f "${EXP_DIR}/env.sh" ]; then
    # shellcheck source=/dev/null
    source "${EXP_DIR}/env.sh"
    echo "Sourced ${EXP_DIR}/env.sh"
else
    echo "No env.sh found in ${EXP_DIR}; using defaults or data/kitti.yaml" >&2
fi

# Ensure WROOT is set (either from env.sh or default to project root)
if [ -z "${WROOT:-}" ]; then
    WROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
    export WROOT
    echo "WROOT not found in env.sh; using ${WROOT}"
fi

run_validator() {
    local model_path="$1"
    local outdir="$2"
    local device_arg="$3"

    mkdir -p "$outdir"
    echo "Validating model: $model_path -> $outdir"
    # choose data config based on DATASET_NAME from env.sh if present
    if [ -n "${DATASET_NAME:-}" ]; then
        # 移除 Non-IID 後綴 (A010, A100 等) 來找到對應的資料集配置
        # 例如: kittiOA010 -> kittiO, bdd100kA010 -> bdd100k
        DATASET_BASE=$(echo "$DATASET_NAME" | sed 's/A[0-9]\{3\}$//')
        DATA_CFG="$WROOT/data/${DATASET_BASE}.yaml"
        if [ ! -f "$DATA_CFG" ]; then
            echo "Warning: Dataset config not found for ${DATASET_BASE}, trying original name: ${DATASET_NAME}" >&2
            DATA_CFG="$WROOT/data/${DATASET_NAME}.yaml"
        fi
    else
        DATA_CFG="$WROOT/data/kitti.yaml"
    fi
    if [ ! -f "$DATA_CFG" ]; then
        echo "Data config $DATA_CFG not found. Please ensure $WROOT/data contains a matching yaml." >&2
        exit 1
    fi
    # ensure baseline is absolute if relative
    if [[ "$BASELINE" != /* ]]; then
        BASELINE="$WROOT/$BASELINE"
    fi
    echo "Using data config: $DATA_CFG"
    echo "Using baseline model: $BASELINE"
    python3 "$PY_VALIDATOR" --device "$device_arg" --experiment-dir "$EXP_DIR" --data-config "$DATA_CFG" --output-dir "$outdir" --baseline-model "$BASELINE"
}

if [ ! -d "$EXP_DIR" ]; then
    echo "Experiment directory not found: $EXP_DIR" >&2
    exit 1
fi

# Prefer aggregated_weights, otherwise check fed_agg_logs for weight files
WEIGHT_DIR="${EXP_DIR}/aggregated_weights"
if [ ! -d "$WEIGHT_DIR" ]; then
    WEIGHT_DIR="${EXP_DIR}/fed_agg_logs"
fi

if [ ! -d "$WEIGHT_DIR" ]; then
    echo "No aggregated_weights or fed_agg_logs folder found in: $EXP_DIR" >&2
    exit 1
fi

RESULTS_DIR="${EXP_DIR}/results"
mkdir -p "$RESULTS_DIR"

# Find weight files matching w_s_r*.pt and sort them by round number
mapfile -t weight_files < <(ls -1 "$WEIGHT_DIR"/w_s_r*.pt 2>/dev/null | sort -V)
if [ ${#weight_files[@]} -eq 0 ]; then
    echo "No weight files (w_s_r*.pt) found in: $WEIGHT_DIR" >&2
    exit 1
fi

echo "Found ${#weight_files[@]} models to validate in $WEIGHT_DIR"

# Save run metadata (dataset, server alg, weights) -> results/run_info.json
export WEIGHT_DIR
export BASELINE
export DATASET_NAME
export SERVER_ALG
export CLIENT_NUM
export TOTAL_ROUNDS
export WROOT
python3 - <<PY > "$RESULTS_DIR/run_info.json"
import json, os, glob, datetime, re
wdir = os.environ['WEIGHT_DIR']
files = sorted(glob.glob(os.path.join(wdir, 'w_s_r*.pt')))
def extract_round_num(filepath):
    fname = os.path.basename(filepath)
    match = re.search(r'w_s_r(\d+)', fname)
    return int(match.group(1)) if match else 0
files = sorted(files, key=extract_round_num)
weights = [{"round": extract_round_num(x), "file": x} for x in files]
info = {
    "dataset_name": os.environ.get('DATASET_NAME', ''),
    "server_alg": os.environ.get('SERVER_ALG', ''),
    "client_num": os.environ.get('CLIENT_NUM', ''),
    "total_rounds": os.environ.get('TOTAL_ROUNDS', ''),
    "baseline": os.environ.get('BASELINE', ''),
    "weight_dir": wdir,
    "weights": weights,
    "wroot": os.environ.get('WROOT', ''),
    "timestamp": datetime.datetime.now().isoformat(),
}
print(json.dumps(info, indent=2))
PY

# Call the Python validator once; it will validate all aggregated weights and save per-model results
run_validator "${weight_files[0]}" "$RESULTS_DIR" "$DEVICE"

echo "Validation run(s) completed."
