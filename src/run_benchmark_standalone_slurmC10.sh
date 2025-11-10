#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
cd $bin/..;
set -u  # Exit on any error

# sbatch ./src/standalone_slurm.sb conf/kittiO/10/fedavg_env.sh
# sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiO/10/fedavgm_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiO/10/fedprox_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiO/10/fednova_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiO/10/fedawa_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiO/10/fedyoga_env.sh

# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/kittiOA010/10/fedavg_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/10/fedavgm_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/10/fedprox_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/10/fednova_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/10/fedawa_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/kittiOA010/10/fedyoga_env.sh
