add_executable(process_sweep_results tracker run_statistics process_sweep_results)
target_link_libraries(process_sweep_results examm_strategy exact_common exact_time_series exact_word_series pthread)

find_package(MPI)

if (MPI_FOUND)
    set (CMAKE_CXX_COMPILE_FLAGS "${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})

    #add_executable(exact_mpi exact_mpi)
    #target_link_libraries(exact_mpi exact_strategy exact_image_tools exact_common ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

    add_executable(examm_mpi examm_mpi)
    target_link_libraries(examm_mpi examm_strategy exact_time_series exact_word_series exact_common ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

    add_executable(examm_mpi_multi examm_mpi_multi)
    target_link_libraries(examm_mpi_multi examm_strategy exact_time_series exact_word_series exact_common ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

    set (CMAKE_CXX_COMPILE_FLAGS "${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})

    add_executable(rnn_kfold_sweep rnn_kfold_sweep)
    target_link_libraries(rnn_kfold_sweep examm_strategy exact_common exact_time_series exact_word_series ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)
endif (MPI_FOUND)
