#!/bin/bash -l
#SBATCH -J neuroevol_random
#SBATCH --account=neuroevolution
#SBATCH --mail-user=slack:@dv6943
#SBATCH --mail-type=ALL
#SBATCH -t 0-12:30:00
#SBATCH --output=%n_%x_%j.out
#SBATCH --error=%x_%a_%j.err
#SBATCH --partition=tier3
#SBATCH --nodes=1			# How many nodes to run on
#SBATCH --ntasks=1			# How many tasks per node
#SBATCH --cpus-per-task=10		# Number of CPUs per task
#SBATCH --mem-per-cpu=10g		# Memory per CPU
#SBATCH --gres=gpu:a100:1

cd /home/dv6943/exact
source ~/new_env/bin/activate
spack load gcc/lhqcen5
spack load libtiff/gnxev37
spack load cmake/pbddesj
spack load openmpi/xcunp5q

sh scripts/backprop_growth_tests/test_random/coal_mpi_random_4_12.sh