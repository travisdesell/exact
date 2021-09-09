add_executable(jordan_rnn jordan_rnn.cxx)
target_link_libraries(jordan_rnn examm_strategy exact_common exact_time_series exact_word_series ${MPI_LIBRARIES} ${MPI_EXTRA} pthread)

add_executable(train_rnn train_rnn.cxx)
target_link_libraries(train_rnn examm_strategy exact_common exact_time_series exact_word_series ${MPI_LIBRARIES} ${MPI_EXTRA} pthread)

#add_executable(rnn_heatmap rnn_heatmap)
#target_link_libraries(rnn_heatmap examm_strategy exact_common exact_time_series exact_word_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} pthread)

add_executable(evaluate_rnn evaluate_rnn.cxx)
target_link_libraries(evaluate_rnn examm_strategy exact_common exact_time_series exact_word_series ${MPI_LIBRARIES} ${MPI_EXTRA} pthread)

add_executable(evaluate_rnns_multi_offset evaluate_rnns_multi_offset.cxx)
target_link_libraries(evaluate_rnns_multi_offset examm_strategy exact_common exact_time_series exact_word_series ${MPI_LIBRARIES} ${MPI_EXTRA} pthread)

add_executable(rnn_statistics rnn_statistics.cxx)
target_link_libraries(rnn_statistics examm_strategy exact_common exact_time_series exact_word_series ${MPI_LIBRARIES} ${MPI_EXTRA} pthread)

