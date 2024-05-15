#!/bin/sh
#
#Testing SIN_Node_GP
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build_laptop/rnn_tests/test_sin_gp_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build_laptop/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type sin_gp --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build_laptop/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 20 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types sin_gp --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing COS_Node_GP
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build_laptop/rnn_tests/test_cos_gp_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build_laptop/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type cos_gp --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build_laptop/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 20 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types cos_gp --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing TANH_Node_GP
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build_laptop/rnn_tests/test_tanh_gp_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build_laptop/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type tanh_gp --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build_laptop/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 20 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types tanh_gp --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing SIGMOID_Node_GP
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build_laptop/rnn_tests/test_sigmoid_gp_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build_laptop/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type sigmoid_gp --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build_laptop/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 20 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_type sigmoid_gp --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing INVERSE_Node_GP
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build_laptop/rnn_tests/test_inverse_gp_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build_laptop/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type inverse_gp --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build_laptop/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 20 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types inverse_gp --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing MULTIPLY_Node_GP
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build_laptop/rnn_tests/test_multiply_gp_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build_laptop/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type multiply_gp --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build_laptop/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 20 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types multiply_gp --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing SUM_Node_GP
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build_laptop/rnn_tests/test_sum_gp_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build_laptop/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type sum_gp --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build_laptop/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 20 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types sum_gp --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +








