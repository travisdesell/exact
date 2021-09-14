#!/bin/bash

for i in {1..10}
do
	./build/multithreaded/examm_mt --number_threads 9 --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 5 --population_size 5 --max_genomes 400 --bp_iterations 10 --output_directory "./test_output" --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level INFO --file_message_level INFO > "./build/output/out_$i.txt"
done
