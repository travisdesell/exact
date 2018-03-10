find_package(MPI)

IF (MPI_FOUND)
    set (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set (CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
    include_directories(${MPI_INCLUDE_PATH})


    add_executable(train_rnn train_rnn one_layer_lstm two_layer_lstm)
    target_link_libraries(train_rnn exalt_strategy exact_common exact_time_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} ${TAO_LIBRARY} pthread)
ENDIF(MPI_FOUND)
