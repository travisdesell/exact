#!/bin/sh

# This is an example of doing island repopulation with EXAMM
# 
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization
# Some repopulation command line arguments:
#   --speciation_method "island": using island speciation method, also the default method. 
#   --extinction_event_generation_number: the frequency of erasing the worst performing island.
#   --islands_to_exterminate: number of islands that is erased at each extinction event.
#   --island_ranking_method "EraseWorst": the only island erasing rule currently. Erase the island whose worst genome is the worst among all islands.
#   --repopulation_method: different ways of repopulating the erased island. can be "bestGenome", "bestParents", "bestIsland", "randomParents".
#   --repopulation_mutations: number of mutations applied to the genome before it getting inserted to the repopulation island
#   --repeat_extinction: allow repeat extinct the same island within certain time if this argument exists. Not allowing repeat extinction gives repopulation islands more time to evolve


cd build

exp_name="../test_output/island_repopulation"
mkdir -p $exp_name
echo "Running island repopulation code with coal dataset, results will be saved to: "$exp_name
echo "###-------------------###"

mpirun -np 8 ./mpi/examm_mpi \
--training_filenames ../datasets/2018_coal/burner_[0-9].csv \
--test_filenames ../datasets/2018_coal/burner_1[0-1].csv \
--time_offset 1 \
--input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
--output_parameter_names Main_Flm_Int \
--number_islands 5 \
--population_size 10 \
--max_genomes 4000 \
--speciation_method "island" \
--extinction_event_generation_number 150 \
--repeat_extinction \
--island_ranking_method "EraseWorst" \
--repopulation_method "bestGenome" \
--islands_to_exterminate 1 \
--repopulation_mutations 2 \
--bp_iterations 1 \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO