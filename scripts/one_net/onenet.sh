#!/bin/sh
# This is an example of running ONENET MPI version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


cd build
# for i in 0 1 2 3 4 5 6 7 8 9
# for i in 0 
# do
    exp_name="../test_output/onenet_mpi"
    mkdir -p $exp_name
    echo "Running base EXAMM code with coal dataset, results will be saved to: "$exp_name
    echo "###-------------------###"

    mpirun -np 16 ./mpi/onenet_mpi \
    --training_filenames ../datasets/2018_coal/burner_[0-9].csv ../datasets/2018_coal/burner_1[0-1].csv \
    --time_offset 1 \
    --speciation_method "onenet" \
    --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
    --output_parameter_names Main_Flm_Int \
    --number_islands 2 \
    --elite_population_size 30 \
    --num_generations 10 \
    --generation_genomes 50 \
    --time_series_length 1000 \
    --bp_iterations 10 \
    --output_directory $exp_name \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --std_message_level ERROR \
    --file_message_level ERROR
# done