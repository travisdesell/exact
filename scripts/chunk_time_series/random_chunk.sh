#!/bin/sh

# This is an example of using uniform random length of time sequence for training genomes with EXAMM
# 
# Some random chunk command line arguments:
#   --random_sequence_length: use uniform random chunk if this argument exists
#   --sequence_length_lower_bound: lower bound for the uniform random chunksize range, 30 if not specified
#   --sequence_length_upper_bound: upper bound for the uniform random chunksize range, 100 if not specified


cd build

MAX_GENOME=4000

echo "\tUsing c172 dateset"
INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM"

exp_name="../test_output/c172_random_chunk"
mkdir -p $exp_name
echo "Running uniform chunk code with coal dataset, results will be saved to: "$exp_name
echo "###-------------------###"

mpirun -np 8 ./mpi/examm_mpi \
--training_filenames ../datasets/2019_ngafid_transfer/c172_file_[1-10].csv \
--test_filenames ../datasets/2019_ngafid_transfer/c172_file_[11-12].csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 10 \
--population_size 10 \
--max_genomes $MAX_GENOME \
--speciation_method "island" \
--random_sequence_length \
--sequence_length_upper_bound 70 \
--extinction_event_generation_number 0 \
--normalize "min_max" \
--bp_iterations 10 \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO
