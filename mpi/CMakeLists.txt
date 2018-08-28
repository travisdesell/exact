find_package(MPI)

if (MPI_FOUND)
    set (CMAKE_CXX_COMPILE_FLAGS "${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})

    add_executable(exact_mpi exact_mpi)
    target_link_libraries(exact_mpi exact_strategy exact_image_tools exact_common ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

    add_executable(exalt_mpi exalt_mpi)
    target_link_libraries(exalt_mpi exalt_strategy exact_time_series exact_common ${MPI_LIBRARIES} ${MPI_EXTRA} ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)
endif (MPI_FOUND)
