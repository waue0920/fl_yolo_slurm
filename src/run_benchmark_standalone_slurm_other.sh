#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
cd $bin/..;
set -u  # Exit on any error

## for foggy dataset
# sbatch ./src/standalone_slurm.sb conf/foggyA010/4/fedavg_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggyA010/4/fedavgm_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggyA010/4/fedprox_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggyA010/4/fednova_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggyA010/4/fedawa_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggyA010/4/fedyoga_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggy/4/fedavg_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggy/4/fedavgm_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggy/4/fedprox_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggy/4/fednova_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggy/4/fedawa_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/foggy/4/fedyoga_env.sh


# ## for sim10k dataset

sbatch ./src/standalone_slurm.sb conf/sim10kA100/4/fedavg_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/sim10kA100/4/fedavgm_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/sim10kA100/4/fedprox_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/sim10kA100/4/fednova_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/sim10kA100/4/fedawa_env.sh
sleep 1m
sbatch ./src/standalone_slurm.sb conf/sim10kA100/4/fedyoga_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/sim10k/4/fedavg_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/sim10k/4/fedavgm_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/sim10k/4/fedprox_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/sim10k/4/fednova_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/sim10k/4/fedawa_env.sh
# sleep 1m
# sbatch ./src/standalone_slurm.sb conf/sim10k/4/fedyoga_env.sh
