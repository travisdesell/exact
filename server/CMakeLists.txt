add_executable(server server)
target_link_libraries(server exact_strategy exact_image_tools ${BOINC_APP_LIBRARIES} ${UNDVC_COMMON_LIBRARY})
