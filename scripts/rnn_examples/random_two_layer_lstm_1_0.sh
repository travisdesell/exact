#!/bin/sh
# This is an example of training fixed two-layer LSTM # This is an example of training fixed two-layer delta network on c172 dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


cd build
INPUT_PARAMETERS="djia"
OUTPUT_PARAMETERS="djia"
DATA_DIR="/home/jefhai/Dropbox/D2S2Lab/Datasets/2020_wind_turbine"

chunk=25


offset=1
num_hidden_layer=2

bp_epoch=1000


for rnn_type in gru
do
    for norm in min_max
    do
        # for index in "${!lower[@]}"
        # do
            for folder in 0 1 2 3 4 5 6 7 8 9
            do
                exp_name="../results/fixed_$rnn_type/wind_${num_hidden_layer}_layer/chunk_$chunk/$folder"
                mkdir -p $exp_name
                echo "Training ${num_hidden_layer}-layer ${rnn_type} network with wind dataset, results will be saved to: "$exp_name
                echo "###-------------------###"

                ./rnn_examples/train_rnn \
                --training_filenames /home/jefhai/Downloads/djia.csv \
                --test_filenames /home/jefhai/Downloads/djia.csv \
                --time_offset $offset \
                --input_parameter_names $INPUT_PARAMETERS \
                --output_parameter_names $OUTPUT_PARAMETERS \
                --bp_iterations $bp_epoch \
                --stochastic \
                --rnn_type $rnn_type \
                --normalize $norm \
                --num_hidden_layers $num_hidden_layer \
                --max_recurrent_depth 10 \
                --output_directory $exp_name \
                --random_sequence_length \
                --sequence_length_lower_bound $chunk \
                --sequence_length_upper_bound $chunk \
                --log_filename fitness.csv \
                --std_message_level ERROR \
                --file_message_level ERROR
            done
        # done
    done
done

