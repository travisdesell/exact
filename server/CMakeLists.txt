add_executable(server server)
target_link_libraries(server evo_cnn_strategy evo_cnn_image_tools ${UNDVC_COMMON_LIBRARY})
