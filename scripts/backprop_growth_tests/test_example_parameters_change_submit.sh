#!/bin/sh
RUNS=${1:-10}

# Global BP bounds
BP_MIN=0
BP_MAX=50

INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int"
OUTPUT_PARAMETERS="Main_Flm_Int"

export INPUT_PARAMETERS
export OUTPUT_PARAMETERS
export BP_MIN
export BP_MAX

echo
echo "Preparing to submit many jobs..."
echo

# =============================
# const: vary bp_iterations
# =============================
cd /home/dv6943/exact/scripts/backprop_growth_tests
jobfile=test_const.sh
CONST_BP_ITERS="0 1 2 4 8 16 32"
for bpiter in $CONST_BP_ITERS; do
  echo "=== const: bp_iterations=$bpiter ==="
  for i in $(seq 1 $RUNS); do
    exp_name="/home/dv6943/exact/test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_${bpiter}/run_${i}"
    mkdir -p "$exp_name"
    echo "Run ${i}/${RUNS}: $exp_name"

    export exp_name
    export bpiter

    jobname=test_const_$bpiter_$i
    sbatch --job-name=$jobname $jobfile

  done
done

# =============================
# rand: vary (bp_min, bp_max)
# =============================
# Pairs: (4,12) (0,16) (12,20) (8,24) (28,36) (24,40)
jobfile=test_random.sh
for pair in "4 12" "0 16" "12 20" "8 24" "28 36" "24 40"; do
  set -- $pair
  bpmin=$1
  bpmax=$2
  echo "=== rand: bp_min=$bpmin, bp_max=$bpmax ==="
  for i in $(seq 1 $RUNS); do
    exp_name="../test_output/line_grid_search/coal_mpi_bp_sweep/rand/bpmin_${bpmin}_bpmax_${bpmax}/run_${i}"
    mkdir -p "$exp_name"
    echo "Run ${i}/${RUNS}: $exp_name"

    export exp_name
    export bpmin
    export bpmax

    jobname=test_random_$bpmin_$bpmax
    sbatch --job-name=$jobname $jobfile

  done
done

# =============================
# scaled: vary (a = slope) and (b = exponent)
# =============================
jobfile=test_scaled.sh
SCALED_A_VALUES="0.0025 0.005 0.01 0.015 0.02"
SCALED_B_VALUES="0.9 0.95 1.0 1.05 1.1"
for a in $SCALED_A_VALUES; do
  for b in $SCALED_B_VALUES; do
    echo "=== scaled: a=$a, b=$b ==="
    for i in $(seq 1 $RUNS); do
        exp_name="../test_output/line_grid_search/coal_mpi_bp_sweep/scaled/a_${a}_b_${b}/run_${i}"
        mkdir -p "$exp_name"
        echo "Run ${i}/${RUNS}: $exp_name"
        export exp_name
        export a
        export b

        jobname=test_scaled_$a_$b
        sbatch --job-name=$jobname $jobfile

    done
  done
done


