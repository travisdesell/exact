add_executable(exact_mpi exact_mpi)
target_link_libraries(exact_mpi exact_strategy exact_image_tools ${UNDVC_COMMON_LIBRARY})
