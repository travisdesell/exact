add_executable(test_net test_net node edge neural_network)

IF (MYSQL_FOUND)
    add_executable(one_layer one_layer)
    target_link_libraries(one_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(two_layer two_layer)
    target_link_libraries(two_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(lenet lenet)
    target_link_libraries(lenet exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)
ELSE (MYSQL_FOUND)
    add_executable(one_layer one_layer)
    target_link_libraries(one_layer exact_strategy exact_common exact_image_tools pthread)

    add_executable(two_layer two_layer)
    target_link_libraries(two_layer exact_strategy exact_common exact_image_tools pthread)

    add_executable(lenet lenet)
    target_link_libraries(lenet exact_strategy exact_common exact_image_tools pthread)
ENDIF (MYSQL_FOUND)
