#!/bin/sh
# This is an example of running EXAMM multithread version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


#cd build

INPUT_PARAMETERS="amp1 amp2 "E1 CHT1" "E1 CHT2" "E1 CHT3" "E1 CHT4" "E1 EGT1" "E1 EGT2" "E1 EGT3" "E1 EGT4"" 
OUTPUT_PARAMETERS="amp1 amp2 "E1 CHT1" "E1 CHT2" "E1 CHT3" "E1 CHT4" "E1 EGT1" "E1 EGT2" "E1 EGT3" "E1 EGT4"" 
#OUTPUT_PARAMETERS="amp1 amp2" 

exp_name="../test_output/ngafid_c712_917"
mkdir -p $exp_name
echo "Running base EXAMM code with SEP2017-NGAFID dataset, results will be saved to: "$exp_name
echo "###-------------------###"

./multithreaded/examm_mt --number_threads 9 \
--training_filenames ../data/ngafid_sens_rnn/train/*.csv --test_filenames \
../data/ngafid_sens_rnn/test/*.csv \
--time_offset 1 \
--cols "E1 CHT1" "E1 CHT2" "E1 CHT3" "E1 CHT4" "E1 EGT1" "E1 EGT2" "E1 EGT3" "E1 EGT4" \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 20 \
--island_size 10 \
--max_genomes 30000 \
--bp_iterations 10 \
--num_mutations 2 \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO
