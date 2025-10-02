#!/bin/bash -l

RUNS=10
BP_MIN=28
BP_MAX=36

cd build

INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int" 
OUTPUT_PARAMETERS="Main_Flm_Int" 

for i in $(seq 1 $RUNS); do
    exp_name="../test_output/new_random_tests/coal_mpi_rand_${BP_MIN}_${BP_MAX}/run_${i}"
    mkdir -p "$exp_name"
    echo "Run ${i}/${RUNS} (bp_iterations=(${BP_MIN}, ${BP_MAX})): results will be saved to: $exp_name"
    echo "###-------------------###"

    srun mpi/examm_mpi \
    --training_filenames ../datasets/2018_coal/burner_[0-9].csv --validation_filenames ../datasets/2018_coal/burner_1[0-1].csv \
    --time_offset 1 \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --number_islands 10 \
    --island_size 10 \
    --max_wallclock_seconds 3600 \
    --backprop_iterations_type "rand" \
    --bp_min $BP_MIN \
    --bp_max $BP_MAX \
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
