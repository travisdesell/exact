add_executable(train_rnn train_rnn)
target_link_libraries(train_rnn examm_strategy exact_common exact_time_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

add_executable(evaluate_rnn evaluate_rnn)
target_link_libraries(evaluate_rnn examm_strategy exact_common exact_time_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} pthread)

add_executable(evaluate_rnns_multi_offset evaluate_rnns_multi_offset)
target_link_libraries(evaluate_rnns_multi_offset examm_strategy exact_common exact_time_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} pthread)

add_executable(rnn_statistics rnn_statistics)
target_link_libraries(rnn_statistics examm_strategy exact_common exact_time_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} pthread)

