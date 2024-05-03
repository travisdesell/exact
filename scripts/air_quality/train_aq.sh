#!/bin/sh
# This is an example of running EXAMM MPI version on pa28 dataset, output parameters are non engine parameters
#
# The pa28 dataset is not normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

cd build

INPUT_PARAMETERS="Date Time PT08.S5(O3) T RH AH"
# OUTPUT_PARAMETERS="CO(GT) NO2(GT) NOx(GT) NMHC(GT)"
OUTPUT_PARAMETERS="CO(GT)"

exp_name="/home/aidan/sandbox/DEEPSPrj/output/init_mvar_1"
mkdir -p "${exp_name}/training"
echo "Running base EXAMM rnn training code with UCI Air Quality dataset, results will be saved to: "$exp_name
echo "###-------------------###"

../../build/rnn_examples/train_rnn \
--training_filenames /home/aidan/sandbox/DEEPSPrj/data/AirQualityUCI.csv \
--test_filenames /home/aidan/sandbox/DEEPSPrj/data/AirQualityUCI.csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--bp_iterations 100000 \
--output_directory "${exp_name}/training" \
--std_message_level INFO \
--file_message_level NONE \
--genome_file $1 \
--learning_rate 0.001 \
--csv_delimiter ";"
