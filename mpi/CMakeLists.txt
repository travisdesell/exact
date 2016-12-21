find_package(MPI REQUIRED)

if (MPI_FOUND)
    set (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set (CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
    include_directories(${MPI_INCLUDE_PATH})

	add_executable(exact_mpi exact_mpi)
	target_link_libraries(exact_mpi exact_common exact_strategy exact_image_tools ${MPI_LIBRARIES} ${MPI_EXTRA} pthread)
endif (MPI_FOUND)
