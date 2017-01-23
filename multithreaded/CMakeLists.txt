<<<<<<< HEAD
add_executable(exact_mt exact_mt)
target_link_libraries(exact_mt exact_strategy exact_image_tools ${UNDVC_COMMON_LIBRARY} pthread)
=======
IF (MYSQL_FOUND)
    add_executable(exact_mt exact_mt)
    target_link_libraries(exact_mt exact_strategy exact_image_tools exact_common ${MYSQL_LIBRARIES} pthread)
ELSE (MYSQL_FOUND)
    add_executable(exact_mt exact_mt)
    target_link_libraries(exact_mt exact_common exact_strategy exact_image_tools pthread)
ENDIF (MYSQL_FOUND)
>>>>>>> dba51fe9e0ad3c6ca2ce5b669d90d058b25b9c39
