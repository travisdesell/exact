#!/bin/sh
# Sweep BP iteration strategies on coal dataset using MPI
# - 1 hour wallclock time per run
# - 10 experiments per setting (can override via first arg)

# Number of runs can be provided as the first argument, defaults to 10
RUNS=${1:-10}

# Global BP bounds
BP_MIN=0
BP_MAX=50

cd build

INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int"
OUTPUT_PARAMETERS="Main_Flm_Int"

# =============================
# const: vary bp_iterations
# =============================
CONST_BP_ITERS="0 1 2 4 8 16 32"
for bp_iter in $CONST_BP_ITERS; do
  echo "=== const: bp_iterations=$bp_iter ==="
  for i in $(seq 1 $RUNS); do
    exp_name="../test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_${bp_iter}/run_${i}"
    mkdir -p "$exp_name"
    echo "Run ${i}/${RUNS}: $exp_name"

    mpirun -np 10 ./mpi/examm_mpi \
      --training_filenames ../datasets/2018_coal/burner_[0-9].csv --validation_filenames ../datasets/2018_coal/burner_1[0-1].csv \
      --time_offset 1 \
      --input_parameter_names $INPUT_PARAMETERS \
      --output_parameter_names $OUTPUT_PARAMETERS \
      --number_islands 10 \
      --island_size 10 \
      --max_wallclock_seconds 3600 \
      --bp_min $BP_MIN \
      --bp_max $BP_MAX \
      --backprop_iterations_type "const" \
      --bp_iterations $bp_iter \
      --output_directory "$exp_name" \
      --num_mutations 2 \
      --weight_update adagrad \
      --eps 0.000001 \
      --beta1 0.99 \
      --sequence_length 50 \
      --possible_node_types simple UGRNN MGU GRU delta LSTM \
      --save_genome_option the_best \
      --std_message_level INFO \
      --file_message_level INFO
  done
done

# =============================
# rand: vary (bp_min, bp_max)
# =============================
# Pairs: (4,12) (0,16) (12,20) (8,24) (28,36) (24,40)
for pair in "4 12" "0 16" "12 20" "8 24" "28 36" "24 40"; do
  set -- $pair
  bpmin=$1
  bpmax=$2
  echo "=== rand: bp_min=$bpmin, bp_max=$bpmax ==="
  for i in $(seq 1 $RUNS); do
    exp_name="../test_output/line_grid_search/coal_mpi_bp_sweep/rand/bpmin_${bpmin}_bpmax_${bpmax}/run_${i}"
    mkdir -p "$exp_name"
    echo "Run ${i}/${RUNS}: $exp_name"

    mpirun -np 10 ./mpi/examm_mpi \
      --training_filenames ../datasets/2018_coal/burner_[0-9].csv --validation_filenames ../datasets/2018_coal/burner_1[0-1].csv \
      --time_offset 1 \
      --input_parameter_names $INPUT_PARAMETERS \
      --output_parameter_names $OUTPUT_PARAMETERS \
      --number_islands 10 \
      --island_size 10 \
      --max_wallclock_seconds 3600 \
      --backprop_iterations_type "rand" \
      --bp_min $bpmin \
      --bp_max $bpmax \
      --output_directory "$exp_name" \
      --num_mutations 2 \
      --weight_update adagrad \
      --eps 0.000001 \
      --beta1 0.99 \
      --sequence_length 50 \
      --possible_node_types simple UGRNN MGU GRU delta LSTM \
      --save_genome_option the_best \
      --std_message_level INFO \
      --file_message_level INFO
  done
done

# =============================
# scaled: vary (a = slope) and (b = exponent)
# =============================
SCALED_A_VALUES="0.0025 0.005 0.01 0.015 0.02"
SCALED_B_VALUES="0.9 0.95 1.0 1.05 1.1"
for a in $SCALED_A_VALUES; do
  for b in $SCALED_B_VALUES; do
    echo "=== scaled: a=$a, b=$b ==="
    for i in $(seq 1 $RUNS); do
      exp_name="../test_output/line_grid_search/coal_mpi_bp_sweep/scaled/a_${a}_b_${b}/run_${i}"
      mkdir -p "$exp_name"
      echo "Run ${i}/${RUNS}: $exp_name"

      mpirun -np 10 ./mpi/examm_mpi \
        --training_filenames ../datasets/2018_coal/burner_[0-9].csv --validation_filenames ../datasets/2018_coal/burner_1[0-1].csv \
        --time_offset 1 \
        --input_parameter_names $INPUT_PARAMETERS \
        --output_parameter_names $OUTPUT_PARAMETERS \
        --number_islands 10 \
        --island_size 10 \
        --max_wallclock_seconds 3600 \
        --bp_min $BP_MIN \
        --bp_max $BP_MAX \
        --backprop_iterations_type "scaled" \
        --bp_slope $a \
        --bp_exponent $b \
        --output_directory "$exp_name" \
        --num_mutations 2 \
        --weight_update adagrad \
        --eps 0.000001 \
        --beta1 0.99 \
        --sequence_length 50 \
        --possible_node_types simple UGRNN MGU GRU delta LSTM \
        --save_genome_option the_best \
        --std_message_level INFO \
        --file_message_level INFO
    done
  done
done


