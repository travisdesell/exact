#!/bin/sh
# This is an example of running EXAMM multithread version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


#cd build

source env_eng.sh

echo "Running base EXAMM code with SEP2017-NGAFID dataset, results will be saved to: "$exp_name
echo "###-------------------###"

make
./multithreaded/examm_mt --number_threads 8 \
--training_filenames $TRAIN_FILES \
--test_filenames ../data/ngafid_sens_rnn/test/*.csv \
--time_offset 1 \
--normalize $NORM_METHOD \
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
