add_executable(evaluate_cnn evaluate_cnn)
target_link_libraries(evaluate_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

IF (MYSQL_FOUND)
    add_executable(one_layer one_layer)
    target_link_libraries(one_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(two_layer two_layer)
    target_link_libraries(two_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(lenet_no_pool lenet_no_pool)
    target_link_libraries(lenet_no_pool exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(large_image_lenet large_image_lenet)
    target_link_libraries(large_image_lenet exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(export_genome export_genome)
    target_link_libraries(export_genome exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(generate_gv generate_gv)
    target_link_libraries(generate_gv exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)
ELSE (MYSQL_FOUND)
    add_executable(one_layer one_layer)
    target_link_libraries(one_layer exact_strategy exact_common exact_image_tools pthread)

    add_executable(two_layer two_layer)
    target_link_libraries(two_layer exact_strategy exact_common exact_image_tools pthread)

    add_executable(lenet_no_pool lenet_no_pool)
    target_link_libraries(lenet_no_pool exact_strategy exact_common exact_image_tools pthread)

    add_executable(large_image_lenet large_image_lenet)
    target_link_libraries(large_image_lenet exact_strategy exact_common exact_image_tools pthread)
ENDIF (MYSQL_FOUND)
