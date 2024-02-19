#!/bin/zsh

let np=8
#SBATCH  --ntasks=8
#SBATCH  --exclude theocho
#SBATCH  --time=8-00:00:00
#SBATCH  -A examm
#SBATCH  --partition=TIER
#SBATCH  -J examm_coal_gp_control
#SBATCH  -o /home/jak5763/exact/results/gp_control/slurm_out/%x.%j.out
#SBATCH  -e /home/jak5763/exact/results/gp_control/slurm_out/%x.%j.err
#SBATCH  --mem=64GB

source lib.zsh

output_dir_prefix=/home/jak5763/exact/results/gp_control
bp_epoch_set=(8)
nfolds=20
MAX_GENOMES=10000
ISLAND_SIZE=10
N_ISLANDS=10
coal
