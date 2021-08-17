add_library(exact_strategy propagation.cxx comparison.cxx pooling.cxx cnn_node.cxx cnn_edge.cxx cnn_genome.cxx exact.cxx)

add_executable(propagation_test propagation.cxx)
target_link_libraries(propagation_test exact_common)
target_compile_definitions(propagation_test PUBLIC -DPROPAGATE_TEST)

add_executable(pooling_test pooling.cxx)
target_link_libraries(pooling_test exact_common)
target_compile_definitions(pooling_test PUBLIC -DPOOL_TEST)

