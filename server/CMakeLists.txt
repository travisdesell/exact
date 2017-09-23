add_executable(exact_bn_sfmp_work_generator exact_work_generator make_jobs)
target_link_libraries(exact_bn_sfmp_work_generator exact_strategy exact_image_tools exact_common ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ${OPENSSL_LIBRARIES} pthread)

add_executable(exact_bn_sfmp_validator
    ${BOINC_INCLUDE_DIR}/sched/validator
    ${BOINC_INCLUDE_DIR}/sched/validate_util
    ${BOINC_INCLUDE_DIR}/sched/validate_util2
    exact_validation_policy)
target_link_libraries(exact_bn_sfmp_validator exact_strategy exact_image_tools exact_common ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ${OPENSSL_LIBRARIRES})

add_executable(exact_bn_sfmp_assimilator
   ${BOINC_INCLUDE_DIR}/sched/assimilator
   ${BOINC_INCLUDE_DIR}/sched/validate_util
   make_jobs
   exact_assimilation_policy)
target_link_libraries(exact_bn_sfmp_assimilator exact_strategy exact_image_tools exact_common ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ${OPENSSL_LIBRARIES})
