#!/bin/sh
# This is a script for evlolving networks to predict Air Quality data
# This also doubles as an example for the csv delimiter option

cd build

INPUT_PARAMETERS="Date Time PT08.S1(CO) PT08.S2(NMHC) PT08.S3(NOx) PT08.S4(NO2) PT08.S5(O3) T RH AH"
OUTPUT_PARAMETERS="CO(GT) NO2(GT) NOx(GT) NMHC(GT)"

exp_name="/home/aidan/sandbox/DEEPSPrj/output/init_mvar_2"
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
--island_size 10 \
--max_genomes 20000 \
--number_threads 14 \
--bp_iterations 15 \
--normalize min_max \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level NONE \
--csv_delimiter ";"
