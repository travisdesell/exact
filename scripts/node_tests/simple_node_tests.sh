#!/bin/sh
#This will do gradient testing on each node, write a genome to binary and then read it into examm_mpi

find ~/exact/scripts/node_tests -type f -name "*.bin" -exec rm {} +

cd ~/exact/build

./rnn_tests/test_sin_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./mpi/examm_mpi --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 8 --bp_iterations 3 --island_size 2 --genome_bin "/Users/jaredmurphy/exact/scripts/node_tests/test_genome.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types simple --std_message_level INFO --file_message_level INFO

find ~/exact/scripts/node_tests -type f -name "*.bin" -exec rm {} +

cd ~/exact


 
