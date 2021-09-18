#!/bin/sh

# This is an example of running NEAT speciation with EXAMM
#
# The NEAT speciation method is implemented based on Stanley's paper:
# Stanley, Kenneth O., and Risto Miikkulainen. "Evolving neural networks through augmenting topologies." Evolutionary computation 10, no. 2 (2002): 99-127.
# 
# The coal dataset used is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization
#
# Some repopulation command line arguments:
#   --speciation_method "neat": using NEAT speciation method. 
#  


cd build

exp_name="../test_output/neat"
mkdir -p $exp_name
echo "Running neat repopulation code with coal dataset, results will be saved to: "$exp_name
echo "###-------------------###"

mpirun -np 8 ./mpi/examm_mpi \
--training_filenames ../datasets/2018_coal/burner_[0-9].csv \
--test_filenames ../datasets/2018_coal/burner_1[0-1].csv \
--time_offset 1 \
--input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
--output_parameter_names Main_Flm_Int \
--number_islands 6 \
--population_size 5 \
--max_genomes 500 \
--speciation_method "neat" \
--species_threshold 3 \
--fitness_threshold 1 \
--neat_c1 1 --neat_c2 1 --neat_c3 0.4 \
--bp_iterations 2 \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO