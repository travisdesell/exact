#!/bin/sh
# This is an example of running EXAMM MPI version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


# Number of runs can be provided as the first argument, defaults to 1
# RUNS=${1:-1}
RUN_NUM=${1:-1}
# Line search values for bp_iterations (space-separated)
BP_INC=${2:-0.0025}
BP_SCALE=${3:-1}

exp_name="../test_output/new/coal_mpi_scaled_linear/bp_inc${BP_INC}_bp_scale${BP_SCALE}/run_${RUN_NUM}"
cd build

INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int" 
OUTPUT_PARAMETERS="Main_Flm_Int" 


echo "=== bp_inc: $BP_INC === bp_scale: $BP_SCALE ==="
mkdir -p "$exp_name"
echo "Run ${RUN_NUM}/10(bp_inc=$BP_INC, bp_scale=$BP_SCALE): results will be saved to: $exp_name"
echo "###-------------------###"

srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive mpi/examm_mpi \
    --training_filenames ../datasets/2018_coal/burner_[0-9].csv --validation_filenames ../datasets/2018_coal/burner_1[0-1].csv \
    --time_offset 1 \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --number_islands 10 \
    --island_size 10 \
    --max_wallclock_seconds 3600 \
    --bp_iterations $BPS \
    --backprop_iterations_type "scaled" \
    --bp_exponent $BP_SCALE \
    --bp_slope $BP_INC \
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

