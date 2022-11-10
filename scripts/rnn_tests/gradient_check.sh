#!/bin/sh
# This is used for backprop gradient checks, make sure forward pass and backward pass work properly

./build/rnn_tests/test_feed_forward_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_lstm_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
./build/rnn_tests/test_jordan_gradients --output_directory results_gradient_check --input_length 20 --std_message_level INFO --file_message_level NONE 
