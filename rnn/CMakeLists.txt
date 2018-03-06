add_library(exalt_strategy bptt rnn_genome lstm_node mse rnn_node rnn_edge rnn_node_interface)

add_executable(lstm_test lstm_node mse rnn_node rnn_node_interface)
target_link_libraries(lstm_test exact_common)
target_compile_definitions(lstm_test PUBLIC -DLSTM_TEST)

add_executable(rnn_genome_test rnn_genome lstm_node mse rnn_node rnn_edge rnn_node_interface)
target_link_libraries(rnn_genome_test exact_common)
target_compile_definitions(rnn_genome_test PUBLIC -DRNN_GENOME_TEST)

#add_executable(pooling_test pooling)
#target_link_libraries(pooling_test exalt_common)
#target_compile_definitions(pooling_test PUBLIC -DPOOL_TEST)

