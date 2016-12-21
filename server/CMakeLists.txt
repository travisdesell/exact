add_executable(exact_work_generator exact_work_generator)
target_link_libraries(exact_work_generator exact_common exact_strategy exact_image_tools ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ${OPENSSL_LIBRARIES} pthread)

add_executable(exact_validator
    ${BOINC_INCLUDE_DIR}/sched/validator
    ${BOINC_INCLUDE_DIR}/sched/validate_util
    ${BOINC_INCLUDE_DIR}/sched/validate_util2
    exact_validation_policy)
target_link_libraries(exact_validator exact_common exact_strategy exact_image_tools ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ${OPENSSL_LIBRARIRES})


