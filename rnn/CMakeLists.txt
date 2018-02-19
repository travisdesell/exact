add_library(exalt_strategy rnn_genome lstm_node rnn_node rnn_edge rnn_node_interface)

add_executable(lstm_test lstm_node rnn_node rnn_node_interface)
target_link_libraries(lstm_test exact_common)
target_compile_definitions(lstm_test PUBLIC -DLSTM_TEST)

#add_executable(pooling_test pooling)
#target_link_libraries(pooling_test exalt_common)
#target_compile_definitions(pooling_test PUBLIC -DPOOL_TEST)

