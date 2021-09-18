#!/bin/bash -l

cd build

exp_name="../test_output/nlp_mt_enarc"
mkdir -p $exp_name
echo "Running nlp code with pennchar dataset, results will be saved to: "$exp_name
echo "###-------------------###"

./multithreaded/examm_nlp --number_threads 9 \
--training_filenames ../datasets/pennchar/train.txt \
--test_filenames ../datasets/pennchar/valid.txt \
--word_offset 1 \
--number_islands 10 \
--population_size 10 \
--max_genomes 2000 \
--bp_iterations 10 \
--output_directory $exp_name \
--possible_node_types simple ENARC \
--normalize min_max \
--std_message_level INFO \
--file_message_level INFO
    