#!/bin/bash -l
#SBATCH -J neuroevol_scexp
#SBATCH --account=neuroevolution
#SBATCH --mail-user=slack:@dv6943
#SBATCH --mail-type=ALL
#SBATCH -t 0-01:30:00
#SBATCH --output=%n_%x_%j.out
#SBATCH --error=%x_%a_%j.err
#SBATCH --partition=tier3
#SBATCH --nodes=1			# How many nodes to run on
#SBATCH --cpus-per-task=1		# Number of CPUs per task
#SBATCH --mem-per-cpu=128g		# Memory per CPU
#SBATCH --gres=gpu:a100:1

cd /home/dv6943/exact
source /home/dv6943/new_env/bin/activate
spack load gcc/lhqcen5
spack load libtiff/gnxev37
spack load openmpi/xcunp5q

sbatch scripts/backprop_growth_tests/general_test/scaled_exp_srun.sh 1 0.0025 0.9
