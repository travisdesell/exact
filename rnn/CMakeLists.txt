add_library(exalt_strategy generate_nn exalt rnn_genome rnn lstm_node gru_node mse rnn_node rnn_edge rnn_recurrent_edge rnn_node_interface)

add_executable(lstm_test lstm_node mse rnn_node rnn_node_interface)
target_link_libraries(lstm_test exact_common)
target_compile_definitions(lstm_test PUBLIC -DLSTM_TEST)


add_executable(gru_test gru_node mse rnn_node rnn_node_interface)
target_link_libraries(gru_test exact_common)
target_compile_definitions(gru_test PUBLIC -DGRU_TEST)

#add_executable(rnn_genome_test rnn_genome rnn lstm_node gru_node mse rnn_node rnn_edge rnn_recurrent_edge rnn_node_interface)
add_executable(rnn_genome_test rnn_genome rnn gru_node lstm_node mse rnn_node rnn_edge rnn_recurrent_edge rnn_node_interface)
target_link_libraries(rnn_genome_test exact_common pthread)
target_compile_definitions(rnn_genome_test PUBLIC -DRNN_GENOME_TEST)

#add_executable(pooling_test pooling)
#target_link_libraries(pooling_test exalt_common)
#target_compile_definitions(pooling_test PUBLIC -DPOOL_TEST)
