#add_executable(exact_mt exact_mt)
#target_link_libraries(exact_mt exact_strategy exact_image_tools exact_common ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

add_executable(exalt_mt_single_series exalt_mt_single_series)
target_link_libraries(exalt_mt_single_series exalt_strategy exact_time_series exact_common pthread)

add_executable(exalt_mt exalt_mt)
target_link_libraries(exalt_mt exalt_strategy exact_time_series exact_common pthread)

add_executable(acnnto_mt acnnto_mt)
target_link_libraries(acnnto_mt acnnto_strategy exact_time_series exact_common pthread)
