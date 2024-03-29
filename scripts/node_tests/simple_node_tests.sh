#!/bin/sh
#
#
#
#

#Testing SIN_Node
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build/rnn_tests/test_sin_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type sin --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 8 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types simple sin --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing SUM_Node
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build/rnn_tests/test_sum_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type sum --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 8 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types simple sum --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing COS_Node
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build/rnn_tests/test_cos_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type cos --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 8 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types simple cos --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing TANH_Node
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build/rnn_tests/test_tanh_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type tanh --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 8 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types simple tanh --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing SIGMOID_Node
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build/rnn_tests/test_sigmoid_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type sigmoid --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 8 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types simple sigmoid --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing INVERSE_Node
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build/rnn_tests/test_inverse_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type inverse --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 8 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types simple inverse --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

#Testing MULTIPLY_Node
find ./ -type f -name "genome_original.bin" -exec rm {} +

./build/rnn_tests/test_multiply_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE

./build/rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type multiply --std_message_level INFO --file_message_level NONE

mpirun -np 3 ./build/mpi/examm_mpi --training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames ./datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 3  --max_genomes 8 --bp_iterations 3 --island_size 2 --genome_bin "genome_original.bin" --transfer_learning_version "v1" --output_directory "./test_output" --possible_node_types simple multiply --std_message_level INFO --file_message_level INFO

find ./ -type f -name "genome_original.bin" -exec rm {} +

