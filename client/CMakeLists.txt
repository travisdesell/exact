add_executable(client client)
target_link_libraries(client evo_cnn_strategy evo_cnn_image_tools ${UNDVC_COMMON_LIBRARY})
