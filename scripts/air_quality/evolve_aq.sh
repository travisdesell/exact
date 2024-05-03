#!/bin/sh
# This is a script for evlolving networks to predict Air Quality data
# This also doubles as an example for the csv delimiter option

cd build

# INPUT_PARAMETERS="Date Time PT08.S1(CO) PT08.S2(NMHC) PT08.S3(NOx) PT08.S4(NO2) PT08.S5(O3) T RH AH"
INPUT_PARAMETERS="PT08.S5(O3) T RH AH"
# OUTPUT_PARAMETERS="CO(GT) NO2(GT) NOx(GT) NMHC(GT)"
OUTPUT_PARAMETERS="CO(GT)"

exp_name="/home/aidan/sandbox/DEEPSPrj/output/univar3"
mkdir -p $exp_name
echo "Running base EXAMM code with UCI Air Quality dataset, results will be saved to: "$exp_name
echo "###-------------------###"

../../build/multithreaded/examm_mt \
--training_filenames /home/aidan/sandbox/DEEPSPrj/data/AirQualityUCI.csv \
--test_filenames /home/aidan/sandbox/DEEPSPrj/data/AirQualityUCI.csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 10 \
--min_recurrent_depth 10 \
--max_recurrent_depth 40 \
--island_size 10 \
--max_genomes 20000 \
--number_threads 14 \
--num_mutations 20 \
--bp_iterations 20 \
--normalize min_max \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level NONE \
--csv_delimiter ";"
