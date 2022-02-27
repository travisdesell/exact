#!/bin/sh
# This is an example of running EXAMM MPI version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

cd $1
make -j32
cd ..

exp_name="/home/josh/Temp/$2"
rm -rf $exp_name
mkdir -p $exp_name
echo "Running base EXAMM code with coal dataset, results will be saved to: "$exp_name
echo "###-------------------###"


mpirun --use-hwthread-cpus -np $6 \
$1/mpi/examm_mpi --number_threads $6 \
--training_filenames ./datasets/2018_coal/burner_[0-9].csv --test_filenames \
./datasets/2018_coal/burner_1[0-1].csv \
--time_offset 1 \
--input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
--output_parameter_names Main_Flm_Int \
--number_islands 1 \
--population_size 2 \
--min_intra_crossover_parents 2 \
--max_intra_crossover_parents 2 \
--min_inter_crossover_parents 2 \
--max_inter_crossover_parents 2 \
--max_time_minutes $3 \
--max_genomes $4 \
--bp_iterations $5 \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO # | pv -a > /dev/null
