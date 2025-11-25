#!/bin/bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
cd $bin/..;
set -u  # Exit on any error


sbatch src/standalone_slurm_replay.sb experiments/30_kittiOA010_fedprox_4C_5R_202510300121/; 
# ./src/validate_models.sh experiments/114_kittiOA010_fedavg_4C_10R_202511121904; 
sleep 1m;
sbatch src/standalone_slurm_replay.sb experiments/57_kittiOA010_fedprox_4C_12R_202511041628/;
# ./src/validate_models.sh experiments/115_kittiOA010_fedavgm_4C_10R_202511121905;
# sleep 1m;
# sbatch src/standalone_slurm_replay.sb experiments/26_kittiOA010_fedavg_4C_5R_202510291305/;
# ./src/validate_models.sh experiments/26_kittiOA010_fedavg_4C_5R_202510291305;
# sleep 1m;
# sbatch src/standalone_slurm_replay.sb experiments/27_kittiOA010_fedavgm_4C_5R_202510291750/;
# ./src/validate_models.sh experiments/27_kittiOA010_fedavgm_4C_5R_202510291750;

# ### Non-IID
# # ## ori
# # # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/55_kittiOA010_fedavg_4C_12R_202511041625; 
# ./src/validate_models.sh experiments/55_kittiOA010_fedavg_4C_12R_202511041625; 
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/56_kittiOA010_fedavgm_4C_12R_202511041626;
# ./src/validate_models.sh experiments/56_kittiOA010_fedavgm_4C_12R_202511041626;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/58_kittiOA010_fednova_4C_12R_202511041629;
# ./src/validate_models.sh experiments/58_kittiOA010_fednova_4C_12R_202511041629;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/57_kittiOA010_fedprox_4C_12R_202511041628;
# ./src/validate_models.sh experiments/57_kittiOA010_fedprox_4C_12R_202511041628;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/59_kittiOA010_fedawa_4C_12R_202511041630;
# ./src/validate_models.sh experiments/59_kittiOA010_fedawa_4C_12R_202511041630;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/60_kittiOA010_fedyoga_4C_12R_202511041631;
# ./src/validate_models.sh experiments/60_kittiOA010_fedyoga_4C_12R_202511041631;
# # # ## yoga
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/26_kittiOA010_fedavg_4C_5R_202510291305;
# ./src/validate_models.sh experiments/26_kittiOA010_fedavg_4C_5R_202510291305;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/27_kittiOA010_fedavgm_4C_5R_202510291750;
# ./src/validate_models.sh experiments/27_kittiOA010_fedavgm_4C_5R_202510291750;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/28_kittiOA010_fednova_4C_5R_202510292015;
# ./src/validate_models.sh experiments/28_kittiOA010_fednova_4C_5R_202510292015;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/30_kittiOA010_fedprox_4C_5R_202510300121;
# ./src/validate_models.sh experiments/30_kittiOA010_fedprox_4C_5R_202510300121;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/31_kittiOA010_fedawa_4C_5R_202510300347;
# ./src/validate_models.sh experiments/31_kittiOA010_fedawa_4C_5R_202510300347;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/32_kittiOA010_fedyoga_4C_5R_202510300613;
# ./src/validate_models.sh experiments/32_kittiOA010_fedyoga_4C_5R_202510300613;

# ./src/validate_models.sh experiments/83_foggyA010_fedavg_4C_10R_202511111624;
# ./src/validate_models.sh experiments/87_foggyA010_fedawa_4C_10R_202511111633;
# ./src/validate_models.sh experiments/88_foggyA010_fedyoga_4C_10R_202511111634;
# ./src/validate_models.sh experiments/89_foggy_fedavg_4C_10R_202511111635;

# # ./src/validate_models.sh experiments/46_kittiO_fedavg_10C_12R_202511032227;
# # ./src/validate_models.sh experiments/47_kittiOA010_fedavg_10C_12R_202511040805;
# # ./src/validate_models.sh experiments/61_kittiO_fedavgm_10C_12R_202511042251;
# # ./src/validate_models.sh experiments/62_kittiO_fedprox_10C_12R_202511042252;
# # ./src/validate_models.sh experiments/63_kittiO_fednova_10C_12R_202511042253;
# # ./src/validate_models.sh experiments/64_kittiO_fedawa_10C_12R_202511042255;
# # ./src/validate_models.sh experiments/65_kittiO_fedyoga_10C_12R_202511042256;
# # ./src/validate_models.sh experiments/66_kittiOA010_fedavgm_10C_12R_202511042257;
# # ./src/validate_models.sh experiments/67_kittiOA010_fedprox_10C_12R_202511042258;
# # ./src/validate_models.sh experiments/68_kittiOA010_fednova_10C_12R_202511042259;
# # ./src/validate_models.sh experiments/69_kittiOA010_fedawa_10C_12R_202511042300;
# # ./src/validate_models.sh experiments/70_kittiOA010_fedyoga_10C_12R_202511042301;

# # # ### IID
# # # # ## ori
# # # sbatch src/standalone_slurm_replay.sb experiments/21_kittiO_fedavg_4C_5R_202510282248;
# ./src/validate_models.sh experiments/21_kittiO_fedavg_4C_5R_202510282248; # not yet
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/22_kittiO_fedavgm_4C_5R_202510290119;
# ./src/validate_models.sh experiments/22_kittiO_fedavgm_4C_5R_202510290119;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/23_kittiO_fednova_4C_5R_202510290346;
# ./src/validate_models.sh experiments/23_kittiO_fednova_4C_5R_202510290346; # not yet
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/24_kittiO_fedprox_4C_5R_202510290612;
# ./src/validate_models.sh experiments/24_kittiO_fedprox_4C_5R_202510290612;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/29_kittiO_fedawa_4C_5R_202510292253;
# ./src/validate_models.sh experiments/29_kittiO_fedawa_4C_5R_202510292253;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/33_kittiO_fedyoga_4C_5R_202510300840;
# ./src/validate_models.sh experiments/33_kittiO_fedyoga_4C_5R_202510300840;
# # # ## yoga
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/49_kittiO_fedavg_4C_12R_202511041134;
# ./src/validate_models.sh experiments/49_kittiO_fedavg_4C_12R_202511041134;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/50_kittiO_fedavgm_4C_12R_202511041459;
# ./src/validate_models.sh experiments/50_kittiO_fedavgm_4C_12R_202511041459;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/52_kittiO_fednova_4C_12R_202511041519;
# ./src/validate_models.sh experiments/52_kittiO_fednova_4C_12R_202511041519;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/52_kittiO_fedprox_4C_12R_202511041542;
# ./src/validate_models.sh experiments/52_kittiO_fedprox_4C_12R_202511041542;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/53_kittiO_fedawa_4C_12R_202511041544;
# ./src/validate_models.sh experiments/53_kittiO_fedawa_4C_12R_202511041544;
# # # sleep 1m;
# # # sbatch src/standalone_slurm_replay.sb experiments/54_kittiO_fedyoga_4C_12R_202511041545;
# ./src/validate_models.sh experiments/54_kittiO_fedyoga_4C_12R_202511041545;
