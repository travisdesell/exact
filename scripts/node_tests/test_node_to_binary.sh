#!/bin/sh
#This will do gradient testing on each node, write a genome to binary and then read it into examm_mpi

find ~/exact/scripts/node_tests -type f -name "*.bin" -exec rm {} +

cd ~/exact/build

#for sin node
./rnn_tests/test_node_to_binary --output_directory results_gradient_check --hidden_node_type sin --std_message_level INFO --file_message_level NONE

find ~/exact/scripts/node_tests -type f -name "*.bin" -exec rm {} +
