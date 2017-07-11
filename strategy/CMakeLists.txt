add_library(exact_strategy propagation pooling cnn_node cnn_edge cnn_genome exact)

add_executable(pooling_test pooling)
target_compile_definitions(pooling_test PUBLIC -DPOOL_TEST)
