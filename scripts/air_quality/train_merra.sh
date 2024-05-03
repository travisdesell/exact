#!/bin/sh
# This is an example of running EXAMM MPI version on pa28 dataset, output parameters are non engine parameters
#
# The pa28 dataset is not normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

cd build

INPUT_PARAMETERS="lon lat lev AIRDENS SO4 SO2 RH PS H O3 T U V"
# OUTPUT_PARAMETERS="CO"
OUTPUT_PARAMETERS="CO SO4 SO2 O3"

exp_name="/home/aidan/sandbox/DEEPSPrj/output/merra/multivar_B"
mkdir -p "${exp_name}/training"
echo "Running base EXAMM rnn training code with UCI Air Quality dataset, results will be saved to: "$exp_name"/training"
echo "###-------------------###"

../../build/rnn_examples/train_rnn \
--training_filenames /home/aidan/sandbox/DEEPSPrj/data/MERRA/merra_post.csv \
--test_filenames /home/aidan/sandbox/DEEPSPrj/data/MERRA/poc_merra_test.csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--bp_iterations 5 \
--output_directory "${exp_name}/training" \
--std_message_level INFO \
--file_message_level NONE \
--genome_file $1 \
--learning_rate 0.01 \
--csv_delimiter ","
