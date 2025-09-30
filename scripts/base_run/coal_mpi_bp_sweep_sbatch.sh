#!/bin/bash -l
#SBATCH -J neuroevol_bp_iterations
#SBATCH --account=neuroevolution
#SBATCH --mail-user=slack:@dv6943
#SBATCH --mail-type=ALL
#SBATCH -t 4-23:00:00
#SBATCH --output=%n_%x_%j.out
#SBATCH --error=%x_%a_%j.err
#SBATCH --partition=tier3
#SBATCH --nodes=1			# How many nodes to run on
#SBATCH --ntasks=18			# How many tasks per node
#SBATCH --cpus-per-task=36		# Number of CPUs per task
#SBATCH --mem-per-cpu=64g		# Memory per CPU
#SBATCH --gres=gpu:a100:1

hostname
# SLURM submission script for remote GPU runs
# Usage:
#   sbatch scripts/base_run/coal_mpi_bp_sweep_sbatch.sh            # default 10 runs/setting
#   sbatch scripts/base_run/coal_mpi_bp_sweep_sbatch.sh -- 5       # 5 runs/setting

# Move to repo root (this script lives in scripts/base_run)
cd /home/dv6943/exact

# Activate environment and load accelerators/toolchain
source ~/new_env/bin/activate
spack load cuda/orevlwf
spack load gcc/lhqcen5

# Delegate to base run script; it enforces 1-hour per experiment via --max_wallclock_seconds 3600
sh /home/dv6943/exact/scripts/base_run/coal_mpi_bp_sweep.sh

# Helpful notes:
# - Submit with: sbatch scripts/base_run/coal_mpi_bp_sweep_sbatch.sh
# - Track jobs: squeue --me   (or: squeue -u <username>)
# - Cancel job: scancel <job_id>


