#!/bin/bash -l
#SBATCH -J neuroevol_random
#SBATCH --account=neuroevolution
#SBATCH --mail-user=slack:@dv6943
#SBATCH --mail-type=ALL
#SBATCH -t 0-01:30:00
#SBATCH --output=%n_%x_%j.out
#SBATCH --error=%x_%a_%j.err
#SBATCH --partition=tier3
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g

cd /home/dv6943/exact
source /home/dv6943/new_env/bin/activate
spack load gcc/lhqcen5
spack load libtiff/gnxev37
spack load openmpi/xcunp5q

sh scripts/backprop_growth_tests/test_random/coal_mpi_random_24_40_1.sh
