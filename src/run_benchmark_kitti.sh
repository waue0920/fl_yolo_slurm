#!/bin/bash
# Benchmark script - explicit list of all dataset x algorithm combinations
# Usage: ./src/run_benchmark.sh
## 已知 fedopt 有數值崩潰問題，暫時先不跑 fedopt
#./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedopt  # 已知有錯誤，聚合後模型會崩潰

set -e  # Exit on any error
# start by 1103 client = 10, ori framework 
./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedavg # 
./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedavg # 


# done by 1101 client = 10, yoga framework
# ./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedavg # done 1101
# ./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedavgm # skip first
# ./src/standalone_orchestrate.sh --dataset kittiO --server-alg fednova # skip first
# ./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedprox # skip first
# ./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedawa # done 1031
# ./src/standalone_orchestrate.sh --dataset kittiO --server-alg fedyoga # done 1101

# ./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedavg # done 1101
#./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedprox # skip
# ./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedawa # done 1031
#./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedyoga # done
#./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fedavgm # done 
#./src/standalone_orchestrate.sh --dataset kittiOA010 --server-alg fednova # skip



#./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedavg
#./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedavgm
#./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fednova
#./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedprox
## ./src/standalone_orchestrate.sh --dataset bdd100kO --server-alg fedawa

#./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedavg
#./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedavgm
#./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fednova
#./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedprox
## ./src/standalone_orchestrate.sh --dataset bdd100kOA010 --server-alg fedawa
