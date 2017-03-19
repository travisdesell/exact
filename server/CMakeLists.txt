add_executable(exact2_work_generator exact_work_generator make_jobs)
target_link_libraries(exact2_work_generator exact_strategy exact_image_tools exact_common ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ${OPENSSL_LIBRARIES} pthread)

add_executable(exact2_validator
    ${BOINC_INCLUDE_DIR}/sched/validator
    ${BOINC_INCLUDE_DIR}/sched/validate_util
    ${BOINC_INCLUDE_DIR}/sched/validate_util2
    boinc_common
    exact_validation_policy)
target_link_libraries(exact2_validator exact_strategy exact_image_tools exact_common ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ${OPENSSL_LIBRARIRES})

add_executable(exact2_assimilator
   ${BOINC_INCLUDE_DIR}/sched/assimilator
   ${BOINC_INCLUDE_DIR}/sched/validate_util
   make_jobs
   boinc_common
   exact_assimilation_policy)
target_link_libraries(exact2_assimilator exact_strategy exact_image_tools exact_common ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ${OPENSSL_LIBRARIES})
