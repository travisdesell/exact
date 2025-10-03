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

CONST_BP_ITERS="0 1 2 4 8 16 32"
for bp_iter in $CONST_BP_ITERS; do
  echo "=== const: bp_iterations=$bp_iter ==="
  for i in $(seq 1 $RUNS); do
    sbatch scripts/backprop_growth_tests/general_test/vanilla_srun.sh $i $bp_iter
  done
done

for pair in "4 12" "0 16" "12 20" "8 24" "28 36" "24 40"; do
  set -- $pair
  bpmin=$1
  bpmax=$2
  echo "=== rand: bp_min=$bpmin, bp_max=$bpmax ==="
  for i in $(seq 1 $RUNS); do
    sbatch scripts/backprop_growth_tests/general_test/random_srun.sh $i $bpmin $bpmax
  done
done

SCALED_A_VALUES="0.0025 0.005 0.01 0.015 0.02"
SCALED_B_VALUES="0.9 0.95 1.0 1.05 1.1"
for a in $SCALED_A_VALUES; do
  for b in $SCALED_B_VALUES; do
    echo "=== scaled: a=$a, b=$b ==="
    for i in $(seq 1 $RUNS); do

        sbatch scripts/backprop_growth_tests/general_test/scaled_srun.sh $i $a $b
    done
  done
done
