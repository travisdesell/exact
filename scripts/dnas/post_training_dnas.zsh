#!/bin/zsh
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
        --train_sequence_length 1000 \
        --validation_sequence_length 100 \
        --crystalize_iters $CRYSTALIZE_ITERS \
        --dnas_k $k

      tail -1 $OUTPUT_DIRECTORY/post_training.csv
}

post_training
