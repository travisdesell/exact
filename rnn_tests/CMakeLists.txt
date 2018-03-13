add_executable(test_rnn test_rnn)
target_link_libraries(test_rnn exalt_strategy exact_common exact_time_series ${MYSQL_LIBRARIES} pthread)
