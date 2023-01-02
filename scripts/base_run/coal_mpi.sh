#!/bin/sh
# This is an example of running EXAMM MPI version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


cd build

INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int" 
OUTPUT_PARAMETERS="Main_Flm_Int" 

exp_name="../test_output/coal_mpi"
mkdir -p $exp_name
echo "Running base EXAMM code with coal dataset, results will be saved to: "$exp_name
echo "###-------------------###"

mpirun -np 4 ./mpi/examm_mpi \
--training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 10 \
--island_size 10 \
--max_genomes 2000 \
--bp_iterations 5 \
--output_directory $exp_name \
--num_mutations 2 \
--weight_update adagrad \
--eps 0.000001 \
--beta1 0.99 \
--sequence_length 50 \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO
