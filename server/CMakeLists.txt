add_executable(exact_work_generator exact_work_generator)
target_link_libraries(exact_work_generator exact_strategy exact_image_tools ${UNDVC_COMMON_LIBRARY} ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ${OPENSSL_LIBRARIES} ${Boost_LIBRARIES} pthread)
