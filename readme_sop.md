# Manual Standard Operating Procedure (SOP)

This document explains how to perform Federated Learning experiments step-by-step manually. This method offers maximum control, facilitating debugging and detailed observation.

## 1. Generate Standard Operating Procedure (SOP) Script

First, use the `orchestrate.sh` script with the `--manual` flag to generate a Shell script containing all necessary commands for the experiment. It is recommended to redirect the output to a file, e.g., `sop.sh`.

### Basic Syntax
```bash
./src/orchestrate.sh <DATASET_NAME> <CLIENT_NUM> <TOTAL_ROUNDS> --manual > sop.sh
```

### Parameters
- `DATASET_NAME`: Dataset name (e.g., kitti, cityscapes)
- `CLIENT_NUM`: Number of clients (e.g., 4)
- `TOTAL_ROUNDS`: Federated learning rounds (e.g., 2, 5)
- `--manual`: **Required flag** to generate manual SOP.
- `--val`: Optional flag; if included, the generated SOP will include model validation steps.

### Examples
```bash
# Generate an SOP for kitti dataset, 4 clients, 2 rounds
./src/orchestrate.sh kitti 4 2 --manual > sop.sh

# Generate an SOP for cityscapes dataset, 8 clients, 3 rounds, including validation steps
./src/orchestrate.sh cityscapes 8 3 --manual --val > cityscapes_sop.sh
```

## 2. Review and Execute SOP Script

Open your newly generated `sop.sh` file with a text editor. You will see all commands organized by steps and rounds.

It is recommended to copy commands line-by-line or block-by-block into your terminal to execute, allowing you to check the output of each step before proceeding to the next.

### Debugging Tips
To debug, you can run the SOP script using `bash -x`, which will show the detailed execution process of each command:
```bash
bash -x sop.sh
```

## 3. Manual Process Overview

A typical SOP flow includes the following steps:

1.  **Set Environment Variables**: Export environment variables required for subsequent commands.
2.  **Data Preparation**: Run `data_prepare.py` to partition the dataset.
3.  **Create Experiment Directory**: Create necessary output folders for this experiment.
4.  **Round 1: Client Training**: Submit training tasks for all clients.
5.  **Round 1: Federated Averaging**: Wait for all clients to finish training, then execute weight aggregation.
6.  ... (Repeat Client Training and Federated Averaging until all rounds are finished) ...
7.  **(Optional) Model Validation**: If the `--val` flag was added when generating the SOP, a model validation step will be included at the end.

By following this procedure, you can precisely control the entire Federated Learning flow.
