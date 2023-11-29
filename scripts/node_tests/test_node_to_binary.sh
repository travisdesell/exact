#!/bin/sh
#This will do a test that converts a genome to binary, imports the genome back from binary as a new object, 
#and checks for forward pass output equivilence and backward pass gradient equivilence between the two  

find ~/exact/scripts/node_tests -type f -name "*.bin" -exec rm {} +

cd ~/exact/build

#for sin node
./rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type sin --std_message_level INFO --file_message_level NONE

find ~/exact/scripts/node_tests -type f -name "*.bin" -exec rm {} +
