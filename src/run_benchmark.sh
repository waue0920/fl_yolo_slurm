#!/bin/bash
# Benchmark script - explicit list of all dataset x algorithm combinations
# Usage: ./src/run_benchmark.sh
# Uncomment additional lines below if you want to include them in the run
set -e  # Exit on any error
./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedavg
./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedavgm
./src/standalone_orchestrate.sh --dataset kittiO --server-alg fednova
./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedprox
./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedopt
# ./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedawa

# ./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedavg
# ./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedavgm
# ./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fednova
# ./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedprox
# ./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedopt
# ./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedawa

# ./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedavg
# ./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedavgm
# ./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedawa
# ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedavg
# ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedavgm
# ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedawa