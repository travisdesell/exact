add_executable(process_sweep_results process_sweep_results)
target_link_libraries(process_sweep_results exalt_strategy acnnto_strategy exact_common exact_time_series pthread)

find_package(MPI)

if (MPI_FOUND)
    set (CMAKE_CXX_COMPILE_FLAGS "${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})

    add_executable(exact_mpi exact_mpi)
    target_link_libraries(exact_mpi exact_strategy exact_image_tools exact_common ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

    add_executable(exalt_mpi exalt_mpi)
    target_link_libraries(exalt_mpi exalt_strategy exact_time_series exact_common ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

    add_executable(acnnto_mpi acnnto_mpi)
    target_link_libraries(acnnto_mpi acnnto_strategy exact_time_series exact_common ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

    add_executable(test_ant_march test_ant_march)
    target_link_libraries(test_ant_march acnnto_strategy exact_time_series exact_common ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)


    set (CMAKE_CXX_COMPILE_FLAGS "${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})

    add_executable(rnn_kfold_sweep rnn_kfold_sweep)
    target_link_libraries(rnn_kfold_sweep exalt_strategy exact_common exact_time_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)
endif (MPI_FOUND)
