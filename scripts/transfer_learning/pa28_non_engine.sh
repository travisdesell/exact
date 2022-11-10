#!/bin/sh
# This is an example of running EXAMM MPI version on pa28 dataset, output parameters are non engine parameters
#
# The pa28 dataset is not normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

cd build

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="Pitch Roll AltMSL IAS LatAc NormAc"

exp_name="../test_output/pa28_non_engine"
mkdir -p $exp_name
echo "Running base EXAMM code with pa28 dataset, results will be saved to: "$exp_name
echo "###-------------------###"

mpirun -np 4 ./mpi/examm_mpi \
--training_filenames ../datasets/2019_ngafid_transfer/pa28_file_[1-9].csv \
--test_filenames ../datasets/2019_ngafid_transfer/pa28_file_1[0-2].csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 10 \
--population_size 10 \
--max_genomes 2000 \
--bp_iterations 5 \
--normalize min_max \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO
