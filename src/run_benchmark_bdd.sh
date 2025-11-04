#!/bin/bash
# Benchmark script - explicit list of all dataset x algorithm combinations
# Usage: ./src/run_benchmark.sh
## 已知 fedopt 有數值崩潰問題，暫時先不跑 fedopt
#./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedopt  # 已知有錯誤，聚合後模型會崩潰

set -e  # Exit on any error
./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedavg
./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedavgm
./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fednova
./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedprox
./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedawa
./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedyoga

# ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedavg
# ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedavgm
# ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fednova
# ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedprox
# ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedawa
# ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedyoga