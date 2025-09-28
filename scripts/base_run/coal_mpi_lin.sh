#!/bin/sh
# This is an example of running EXAMM MPI version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


# Number of runs can be provided as the first argument, defaults to 1
RUNS=${1:-1}

# Grid search parameters
BP_SCALES="0.3 0.5 0.7 1.5 2.0"
BP_slope="50 150 250 500 1000"

cd build

INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int" 
OUTPUT_PARAMETERS="Main_Flm_Int" 

for bp_scale in $BP_SCALES; do
  for bp_inc in $BP_slope; do
    echo "=== Grid Search: bp_iterations=$bp_iter, bp_exponent=$bp_exponent, bp_slope=$bp_inc ==="
    for i in $(seq 1 $RUNS); do
      exp_name="../test_output/line_grid_search/coal_mpi_lin/bp_iter_${bp_iter}_scale_${bp_exponent}_inc_${bp_inc}/run_${i}"
      mkdir -p "$exp_name"
      echo "Run ${i}/${RUNS}: results will be saved to: $exp_name"
      echo "###-------------------###"

      mpirun -np 10 ./mpi/examm_mpi \
      --training_filenames ../datasets/2018_coal/burner_[0-9].csv --validation_filenames ../datasets/2018_coal/burner_1[0-1].csv \
      --time_offset 1 \
      --input_parameter_names $INPUT_PARAMETERS \
      --output_parameter_names $OUTPUT_PARAMETERS \
      --number_islands 10 \
      --island_size 10 \
      --max_genomes 2000 \
      --max_wallclock_seconds 1500 \
      --backprop_iterations_type "linear" \
      --bp_exponent $bp_scale \
      --bp_slope $bp_inc \
      --bp_max 50 \
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
