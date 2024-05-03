#!/bin/sh
# This is a script for evlolving networks to predict Air Quality data
# This also doubles as an example for the csv delimiter option

cd build

INPUT_PARAMETERS="lon lat lev AIRDENS SO4 SO2 RH PS H O3 T U V"
# OUTPUT_PARAMETERS="CO"
OUTPUT_PARAMETERS="CO SO4 SO2 O3"

exp_name="/home/aidan/sandbox/DEEPSPrj/output/merra/mv-A"
mkdir -p $exp_name
echo "Running base EXAMM code with MERRA-2 dataset, results will be saved to: "$exp_name
echo "###-------------------###"

../../build/multithreaded/examm_mt \
--training_filenames /home/aidan/sandbox/DEEPSPrj/data/MERRA/merra_100k_23.csv \
--test_filenames /home/aidan/sandbox/DEEPSPrj/data/MERRA/merra_100k_23_test.csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 10 \
--min_recurrent_depth 1 \
--max_recurrent_depth 100 \
--island_size 10 \
--max_genomes 1000 \
--number_threads 14 \
--num_mutations 20 \
--bp_iterations 5 \
--normalize none \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level NONE \
--csv_delimiter ","
