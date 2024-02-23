#!/bin/sh
# This is used for backprop gradient checks, make sure forward pass and backward pass work properly

./build/rnn_tests/test_feed_forward_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_jordan_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_elman_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 

./build/rnn_tests/test_lstm_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_ugrnn_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_delta_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_mgu_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_gru_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_enarc_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_enas_dag_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_random_dag_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 

./build/rnn_tests/test_sin_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_sum_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE
./build/rnn_tests/test_cos_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE
./build/rnn_tests/test_tanh_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE
./build/rnn_tests/test_sigmoid_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE
./build/rnn_tests/test_inverse_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE




 
 
