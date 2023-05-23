source env_eng.sh

epochs=$1
genome=$2
echo "Training for ${epochs} epochs, results saved to: ${exp_name}"
echo "Using ${TRAIN_FILES} and ${TEST_FILES}"

exp_name=$exp_name/train
mkdir -p $exp_name

../../build/rnn_examples/train_rnn \
--training_filenames $TRAIN_FILES \
--test_filenames $TEST_FILES \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--bp_iterations $epochs \
--genome_file $genome \
--stochastic \
--normalize $NORM_METHOD \
--output_directory $exp_name \
--random_sequence_length \
--log_filename fitness.csv \
--learning_rate 0.001 \
--std_message_level INFO \
--file_message_level ERROR

#--sequence_length_lower_bound ${lower[$index]} \
#--sequence_length_upper_bound ${upper[$index]} \
