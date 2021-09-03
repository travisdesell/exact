#!/bin/sh
# This is an example of training fixed two-layer ugrnn # This is an example of training fixed two-layer delta network on c172 dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization


cd build

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM"

lower=(50 20 100 200 500 1000 2000 50 50 50)
upper=(75 100 200 500 1000 2000 5000 200 500 1000)

offset=40
num_hidden_layer=2

bp_epoch=1000


for rnn_type in ugrnn
do
    for norm in min_max
    do
        for index in "${!lower[@]}"
        do
            for folder in 10 11 12 13 14 15 16 17 18 19
            do
                exp_name="../results/aaai_random_length/c172_${num_hidden_layer}_layer/$rnn_type/offset_$offset/uniform_${lower[$index]}_${upper[$index]}/$norm/$folder"
                mkdir -p $exp_name
                echo "Training ${num_hidden_layer}-layer ugrnn network with c172 dataset, results will be saved to: "$exp_name
                echo "###-------------------###"

                ./rnn_examples/train_rnn \
                --training_filenames ../datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
                --test_filenames ../datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
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
                --sequence_length_lower_bound ${lower[$index]} \
                --sequence_length_upper_bound ${upper[$index]} \
                --log_filename fitness.csv \
                --learning_rate 0.0001 \
                --std_message_level ERROR \
                --file_message_level ERROR
            done
        done
    done
done

