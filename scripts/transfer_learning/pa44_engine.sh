#!/bin/sh
# This is an example of running EXAMM MPI version on pa44 dataset, output parameters are engine parameters
#
# The pa44 dataset is not normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

cd build

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_MAP E1_OilP E1_OilT E1_RPM E2_CHT1 E2_EGT1 E2_EGT2 E2_EGT3 E2_EGT4 E2_FFlow E2_OilP E2_OilT E2_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="E1_CHT1 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_MAP E1_OilP E1_OilT E1_RPM E2_CHT1 E2_EGT1 E2_EGT2 E2_EGT3 E2_EGT4 E2_FFlow E2_OilP E2_OilT E2_RPM"

exp_name="../test_output/pa44_engine"
mkdir -p $exp_name
echo "Running base EXAMM code with pa44 dataset, results will be saved to: "$exp_name
echo "###-------------------###"

mpirun -np 4 ./mpi/examm_mpi \
--training_filenames ../datasets/2019_ngafid_transfer/pa44_file_[1-9].csv \
--test_filenames ../datasets/2019_ngafid_transfer/pa44_file_1[0-2].csv \
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
