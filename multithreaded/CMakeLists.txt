add_executable(exact_mt exact_mt)
target_link_libraries(exact_mt exact_strategy exact_image_tools ${UNDVC_COMMON_LIBRARY} pthread)
