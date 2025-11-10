#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
cd $bin/..;
set -u  # Exit on any error

sbatch ./src/standalone_slurm.sb conf/kittiOA010/4/fedavg_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/4/fedavgm_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/4/fedprox_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/4/fednova_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/4/fedawa_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/4/fedyoga_env.sh
