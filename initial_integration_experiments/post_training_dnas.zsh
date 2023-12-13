#!/usr/bin/zsh
INPUT_PARAMETERS='AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd'
OUTPUT_PARAMETERS='E1_EGT1'

offset=1

post_training() {

    echo "genome = $GENOME"
    Release/rnn_examples/train_rnn \
        --training_filenames datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
        --test_filenames datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
        --time_offset 1 \
        --input_parameter_names ${=INPUT_PARAMETERS} \
        --output_parameter_names ${=OUTPUT_PARAMETERS} \
        --bp_iterations $BP_ITERS \
        --stochastic \
        --normalize min_max \
        --genome_file $GENOME \
        --output_directory $OUTPUT_DIRECTORY \
        --log_filename post_training.csv \
        --learning_rate 0.01 \
        --weight_update adagrad \
        --train_sequence_length 100 \
        --validation_sequence_length 100 \
        --crystalize_iters $CRYSTALIZE_ITERS \
        --dnas_k $k
 
}

post_training
