#!/bin/bash

# ===================================================================================
# Decoupled Executor Script for a Single Client Training Task
# ===================================================================================
# This script is a self-contained training executor. It is called by a Slurm
# script (like client_train.sb) and is responsible for executing the
# YOLOv9 training in a Singularity container.
#
# It is fully parameterized and does not depend on environment variables like
# EXP_ID or ROUND_NUM. All paths and hyperparameters are passed as arguments.
# ===================================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- 0. Argument Parsing ---
# Initialize variables
DATA_YAML=""
WEIGHTS_IN=""
PROJECT_OUT=""
NAME_OUT=""
EXTRA_ARGS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data-yaml) DATA_YAML="$2"; shift ;;
        --weights-in) WEIGHTS_IN="$2"; shift ;;
        --project-out) PROJECT_OUT="$2"; shift ;;
        --name-out) NAME_OUT="$2"; shift ;;
        --extra-args) EXTRA_ARGS="$2"; shift ;; # Pass extra arguments as a single string
        *) echo "Unknown parameter passed: $1"; exit 1 ;; # Handle unknown parameters
    esac
    shift
done

# Verify required arguments
if [ -z "${DATA_YAML}" ] || [ -z "${WEIGHTS_IN}" ] || [ -z "${PROJECT_OUT}" ] || [ -z "${NAME_OUT}" ]; then
    echo "Usage: $0 \
    --data-yaml <path_to_client.yaml> \
    --weights-in <path_to_input_weights.pt> \
    --project-out <path_to_output_project_dir> \
    --name-out <output_run_name> \
    [--extra-args \"--epochs 50 --batch 8\"]"
    exit 1
fi

# --- 1. Project Root and Environment Setup ---
# Use WROOT environment variable (set by user before execution)
if [ -z "${WROOT}" ]; then
    echo "Error: WROOT environment variable is not set"
    echo "Please run: export WROOT=/path/to/project/root"
    exit 1
fi

SINGULARITY_IMG="${WROOT}/yolo9t2_ngc2306_20241226.sif"
MODEL_CFG="${WROOT}/yolov9/models/detect/yolov9-c.yaml"

echo "================================================================================"
echo ">> Starting Client Training (Decoupled)"
echo ">> Project Root:    ${WROOT}"
echo ">> Data YAML:       ${DATA_YAML}"
echo ">> Input Weights:   ${WEIGHTS_IN}"
echo ">> Output Project:  ${PROJECT_OUT}"
echo ">> Output Name:     ${NAME_OUT}"
echo ">> Extra Args:      ${EXTRA_ARGS}"
echo "================================================================================"

# Check if files exist before proceeding
if [ ! -f "${WEIGHTS_IN}" ]; then
    echo "Error: Input weights file not found at ${WEIGHTS_IN}"
    exit 1
fi
if [ ! -f "${WROOT}/${DATA_YAML}" ]; then
    echo "Error: Data YAML file not found at ${WROOT}/${DATA_YAML}"
    exit 1
fi
if [ ! -f "${SINGULARITY_IMG}" ]; then
    echo "Error: Singularity image not found at ${SINGULARITY_IMG}"
    exit 1
fi

# --- 2. Execute YOLOv9 Training inside Singularity ---
echo ">> Executing training command inside Singularity container..."

# Note: All paths passed to train_dual.py must be from the perspective of
# inside the container, which is the same as the host system due to the bind mount.
singularity exec --nv \
    --bind ${WROOT}:${WROOT} \
    --bind /home/waue0920/dataset/yolo:/home/waue0920/dataset/yolo \
    "${SINGULARITY_IMG}" \
python3 "${WROOT}/yolov9/train_dual.py" \
    --weights "${WEIGHTS_IN}" \
    --data "${WROOT}/${DATA_YAML}" \
    --cfg "${MODEL_CFG}" \
    --project "${PROJECT_OUT}" \
    --name "${NAME_OUT}" \
    ${EXTRA_ARGS} # Pass all other hparams

EXIT_CODE=$?
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: YOLOv9 training failed with exit code ${EXIT_CODE}."
    exit ${EXIT_CODE}
fi

echo "================================================================================"
echo ">> Client training finished successfully."
echo "================================================================================"
