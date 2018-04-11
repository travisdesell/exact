add_executable(exact_mt exact_mt)
target_link_libraries(exact_mt exact_strategy exact_image_tools exact_common ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

add_executable(exalt_mt exalt_mt)
target_link_libraries(exalt_mt exalt_strategy exact_time_series exact_common pthread)
