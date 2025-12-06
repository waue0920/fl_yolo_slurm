# YOLOv9 Federated Learning Framework

A Federated Learning system based on YOLOv9, utilizing the Slurm cluster environment for distributed training and aggregation.

[**中文說明 (Chinese)**](README_zh.md)

---

## 📋 Quick Guide

### 1. Introduction
- [System Overview](#-system-overview)
- [Requirements](#-requirements)
- [Installation & Setup](#-installation--setup)
- [Directory Structure](#-directory-structure)

### 2. Execution
- [Quick Start (Auto Mode)](#-quick-start)
- [Replay Experiment](#-replay-experiment)
- [Standalone Mode (No Slurm)](#-standalone-mode)
- [Unit Tests](#-unit-tests)

### 3. Validation & Extras
- [Model Validation](#-model-validation)
- [Monitoring & Debugging](#-monitoring--debugging)
- [Supported Algorithms](#-supported-algorithms)

---

## 🎯 System Overview

This framework is designed to implement a complete Federated Learning (FL) workflow using the NCHC HPC cluster environment (TWCC / N5).

The project uses initial model weights (`yolov9-c.pt`) for pre-training. It distributes these weights to multiple Clients. Each Client trains on its own data subset and sends the updated weights back to the Server for aggregation (Federated Averaging) to produce a new global model for the next round. This process repeats for multiple rounds to train a high-performance global model while preserving data privacy.

* Federated Learning Workflow:
```
Round 1: yolov9-c.pt → [Client1, Client2, Client3, Client4] → w_s_r1.pt
Round 2: w_s_r1.pt  → [Client1, Client2, Client3, Client4] → w_s_r2.pt
Round 3: w_s_r2.pt  → [Client1, Client2, Client3, Client4] → w_s_r3.pt
...
```
![FL Workflow](pics/fl_hpc_overview.gif)

---

## 🛠️ Requirements
- **Execution Environment**: NCHC [TWCC](https://www.nchc.org.tw/Page?itemid=6&mid=10)
  - **OS**: Linux
  - **Scheduler**: Slurm Workload Manager
  - **Container Engine**: Singularity
  - **Python**: ≥ 3.8
  - **GPU**: NVIDIA GPU (CUDA supported)
  - **PyTorch**: PyTorch (≥ 2.1.0)
  - **Experiment Tracking**: Wandb

---

## 📦 Installation & Setup

### 1. Clone Project & Submodules
```bash
# Clone the main repository
git clone <repository-url>
cd fl_yolo_slurm

# Download yolov9 submodule
# (Or manually: git clone https://github.com/WongKinYiu/yolov9.git)
git submodule update --init --recursive
```

### 2. Prepare Necessary Files
Ensure the following files are in the project root:
- **Singularity Image**: `yolo9t2_ngc2306_20241226.sif` ([twcc-cos download](https://cos.twcc.ai/wauehpcproject/yolo9t2_ngc2306_20241226.sif))
- **Initial Weights**: `yolov9-c.pt` ([official download](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt))

### 3. Prepare Datasets
Place your datasets in the `datasets/` directory and create corresponding `.yaml` config files in `data/`.
- **[📖 Dataset Preparation Guide](./readme_datasets.md)**

### 4. Directory Structure
```
.
├── README.md               # Documentation (English)
├── README_zh.md            # Documentation (Chinese)
├── readme_sop.md           # 📖 Manual SOP Guide
├── readme_val.md           # 📊 Validation Guide
├── readme_debug.md         # 🔍 Debugging Guide
├── yolov9/                 # YOLOv9 Source (Git Submodule)
├── src/                    # Source Code
│   ├── orchestrate.sh      # Main Script
│   └── ...                 # Helper Scripts
├── data/                   # Dataset YAML Configs
├── datasets/               # Raw Datasets
├── federated_data/         # Partioned Client Data
├── experiments/            # Experiment Outputs
│   └── {EXP_ID}/
├── yolo9t2_ngc2306_20241226.sif    # Singularity Container
└── yolov9-c.pt             # Initial Pre-trained Weights
```

---

## 🚀 Quick Start

### Fully Automated Mode (Slurm Cluster)
```bash
# Method 1: Using sbatch (All jobs run on worker nodes)
sbatch src/run.sb 

# Method 2: Using orchestrate.sh (Orchestrator runs on login node)
./src/orchestrate.sh kitti 4 2
```
> **Tip**: To include final model validation, add the `--val` flag.

The execution will automatically detect if dataset partitioning is needed, then launch n+1 Slurm processes:
- n client training jobs (parallel)
- 1 server aggregation job (waiting for clients to complete)

![slurm](pics/sim10k_c4_r5_slurm.png)

---

## 🔄 Replay Experiment

If a Federated Learning experiment fails mid-way, you can use `replay.sh` to resume from the breaking point:

```bash
# Specify experiment directory, automatically detects finished rounds and resumes
./src/replay.sh experiments/18_kitti_fedavg_4C_6R_202510010849
```

**Features**:
- ✅ Automatically detects finished rounds
- ✅ Resumes from failure point
- ✅ Complete logging
- ✅ Avoids re-running completed rounds

---

## 💻 Standalone Mode (No Slurm)

Suitable for local environments or servers without Slurm:

```bash
# Parameters are read from env.sh
./src/standalone_orchestrate.sh

# Dry-run mode: Show commands without execution
./src/standalone_orchestrate.sh --dry-run
```

**Features**:
- ✅ No Slurm required
- ✅ Sequential execution of client training
- ✅ Good for small-scale testing and debugging
- ✅ Supports dry-run preview

---

## 🧪 Unit Tests

Quickly test aggregation algorithms and training flows:

```bash
# Edit src/run_unit_test.sh to set EXP_ID and algorithm
# Then run:
./src/run_unit_test.sh
```

**Test Process**:
1. Use existing Round 1 client outputs for aggregation test
2. Use aggregated weights for Round 2 client training test
3. Verify loss values and NaN/Inf handling

---

### 📖 Manual SOP (Advanced)
**Note**: Manual SOP mode is deprecated in v3 in favor of automation.
For historical reference or detailed step-by-step breakdown, please refer to:
- **[📖 Manual SOP Guide](./readme_sop.md)**

For current detailed control, use `standalone_orchestrate.sh` or refer to the scripts directly.

---

## 📊 Model Validation

The system provides complete analysis of FL model performance.
- **[📊 Model Validation Guide](./readme_val.md)**

![validation](pics/kitti_c4_r3_val.png)

---

## 🔍 Monitoring & Debugging

Provides Slurm monitoring, log checking, and solutions for common issues.
- **[🔍 Monitoring & Debugging Guide](./readme_debug.md)**

---

## 🧮 Supported Algorithms

Configure `SERVER_ALG` in `src/env.sh`:

| Algorithm | Description | Scenario | Hyperparameters |
|-----------|-------------|----------|-----------------|
| **fedavg** | Federated Averaging | General, IID Data | - |
| **fedprox** | FedProx | Non-IID Data | `SERVER_FEDPROX_MU` |
| **fedavgm** | FedAvgM (Server Momentum) | Faster Convergence | `SERVER_FEDAVGM_LR`, `SERVER_FEDAVGM_MOMENTUM` |
| **fedopt** | FedOpt (Server Adam) | Stable Training | `SERVER_FEDOPT_LR`, `SERVER_FEDOPT_BETA1`, `SERVER_FEDOPT_BETA2` |
| **fedyoga** | **FedYOGA (Adaptive)** | **Non-IID, Imbalanced** | `SERVER_FEDYOGA_PCA_DIM`, `SERVER_FEDYOGA_CLIP_THRESHOLD`, etc. |
| **fednova** | FedNova | Heterogeneous Steps | `SERVER_FEDNOVA_MU`, `SERVER_FEDNOVA_LR` |

### FedYOGA Features

**FedYOGA** is an advanced aggregation algorithm optimized for Non-IID and imbalanced data:

- ✅ **PCA Dimensionality Reduction**: Reduces weight variance dimensions for efficient aggregation.
- ✅ **Adaptive Weights**: Dynamically adjusts weights based on client loss drop and gradient variance.
- ✅ **Numerical Stability**:
  - Auto-detects and repairs NaN/Inf in BatchNorm statistics.
  - Skips corrupted client weights.
  - Weight difference clipping.
- ✅ **Complexity Analysis**: Auto-calculates space and communication complexity.

**Configuration Example** (`src/env.sh`):
```bash
export SERVER_ALG="fedyoga"
export SERVER_FEDYOGA_HISTORY_WINDOW=5
export SERVER_FEDYOGA_PCA_DIM=4
export SERVER_FEDYOGA_CLIP_THRESHOLD=10.0
```

---

## 🛡️ Error Handling & Stability

### NaN/Inf Auto-Repair
1. **BatchNorm Repair**: Resets corrupted `running_mean`, `running_var`.
2. **Critical Parameter Detection**: Skips clients with NaN/Inf weights.
3. **Diagnostics**: Provides possible causes and solutions.

### Dynamic Port Allocation
Avoids NCCL port conflicts during parallel client training:
- Automatically finds available ports (10000-60000).
- Assigns unique ports to each client.

### Experiment Logs
All outputs are logged to `experiments/{EXP_ID}/orchestrator.log`.

---

**Last Updated**: 2025-10-20
**Version**: v3.0 (Stability Enhanced)
**Maintainer**: nchc/waue0920

---

## 📸 Snapshots

### 1. Validation Result (Cityscapes, 4 Clients, 5 Rounds)
![Validation on Cityscapes](pics/cityscape_c4_r5_val.jpg)

### 2. Training Metrics (By Round)
![Metrics by Round](pics/cityscape_c4_r5_e50_byRound.png)

### 3. Training Metrics (By Epoch)
![Metrics by Epoch](pics/cityscape_c4_r5_e50_byEpoch.png)

### 4. Wandb Dashboard
![Wandb Dashboard](pics/cityscape_c4_r5_e50_Wandb.png)
