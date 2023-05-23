#!/bin/sh
# This is an example of running EXAMM multithread version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


#cd build

INPUT_PARAMETERS="E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM"  
OUTPUT_PARAMETERS="E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM" 

#
echo $INPUT_PARAMETERS

exp_name="../test_output/ngafid_c712_917"
mkdir -p $exp_name
echo "Running base EXAMM code with SEP2017-NGAFID dataset, results will be saved to: "$exp_name
echo "###-------------------###"

make
./multithreaded/examm_mt --number_threads 8 \
--training_filenames ../data/ngafid_sens_rnn/train/*.csv \
--test_filenames ../data/ngafid_sens_rnn/test/*.csv \
--time_offset 1 \
--normalize min_max \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 10 \
--island_size 10 \
--max_genomes 30000 \
--bp_iterations 5 \
--num_mutations 4 \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO
