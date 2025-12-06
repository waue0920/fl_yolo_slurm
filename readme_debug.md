# Monitoring and Debugging Guide

This document provides guidelines for system monitoring, log checking, and troubleshooting when running Federated Learning experiments.

## 1. Slurm Job Monitoring

After submitting an experiment, you can use standard Slurm commands to monitor job status.

```bash
# View the status of all your current jobs (Pending, Running)
squeue -u $USER

# View detailed information of a specific job (e.g., resource allocation, run time)
scontrol show job <JOB_ID>

# Cancel a running or pending job
scancel <JOB_ID>
```

## 2. Log Checking

Logs are the most important source for debugging. In this system, logs are distributed within the experiment directory, each serving a specific purpose.

```bash
# 1. View the full workflow log of the main orchestrator script (orchestrate.sh)
tail -f experiments/{EXP_ID}/orchestrator.log

# 2. View the Slurm output log for a specific client (e.g., client 1, round 1)
#    This contains detailed screen output during YOLOv9 training
tail -f experiments/{EXP_ID}/slurm_logs/client_1_round_1.out

# 3. View the execution log of Federated Averaging (server_fedagg.py)
tail -f experiments/{EXP_ID}/fed_avg_logs/round_1.out
```

## 3. Critical Path and File Checks

When encountering issues, manually check if the output in the following critical paths meets expectations.

```bash
# Check if the dataset has been successfully partitioned for clients
# You should see files like c1.yaml, c1/, c2.yaml, c2/ ...
ls -la federated_data/{DATASET_NAME}_{CLIENT_NUM}/

# Check if aggregated weight files for each round have been generated
ls -la experiments/{EXP_ID}/aggregated_weights/

# Check specific client training outputs, especially if the weights file 'best.pt' exists
ls -la experiments/{EXP_ID}/client_outputs/round_1/client_*/weights/
```

## 4. Common Errors and Solutions

### Error 1: Environment Variable Not Set

- **Error Message**: `Error: WROOT environment variable is not set`
- **Cause**: When manually executing some helper scripts (like `client_train.sh`), the environment variable for the project root directory was not pre-set.
- **Solution**: Manually export the necessary environment variables before running the script.
  ```bash
  # Set WROOT to the absolute path of your project
  export WROOT=$(pwd)
  
  # Check if set successfully
  echo $WROOT
  ```
  > **Note**: You usually won't encounter this issue when using the main script `orchestrate.sh`, as it handles this automatically.

### Error 2: Singularity Image Path Error

- **Error Message**: `Error: Singularity image not found`
- **Cause**: The script cannot find the container image file at the default path (`${WROOT}/yolo9t2_ngc2306_20241226.sif`).
- **Solution**: Ensure your `.sif` file actually exists in the project root directory.
  ```bash
  ls -la ${WROOT}/yolo9t2_ngc2306_20241226.sif
  ```

### Debugging Tip: Using `bash -x` Mode

When using a manually generated `sop.sh` script, if you want to see the detailed content of each command before execution (variables will be substituted with actual values), you can use `bash -x` to run it. This is very useful for tracking variable settings or path issues.

```bash
# First generate the SOP script
./orchestrate.sh kitti 4 2 --manual > debug_sop.sh

# Execute using -x mode, detailed command execution process will be printed on screen
bash -x debug_sop.sh
```
