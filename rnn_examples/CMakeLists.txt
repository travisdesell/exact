add_executable(train_rnn train_rnn)
target_link_libraries(train_rnn exalt_strategy exact_common exact_time_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

add_executable(evaluate_rnn evaluate_rnn)
target_link_libraries(evaluate_rnn exalt_strategy exact_common exact_time_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} pthread)
