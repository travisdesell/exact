IF (MYSQL_FOUND)
    add_executable(evaluate_population evaluate_population)
    target_link_libraries(evaluate_population exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    #add_executable(one_layer_lstm one_layer_lstm)
    #target_link_libraries(one_layer_lstm exact_strategy exact_common exact_time_series ${MYSQL_LIBRARIES} pthread)

    add_executable(check_gradients check_gradients)
    target_link_libraries(check_gradients exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(retrain_cnn retrain_cnn)
    target_link_libraries(retrain_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(evaluate_large_image_cnn evaluate_large_image_cnn)
    target_link_libraries(evaluate_large_image_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(evaluate_cnn evaluate_cnn)
    target_link_libraries(evaluate_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    #add_executable(test_exact_database_export test_exact_database_export)
    #target_link_libraries(test_exact_database_export exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(test_genome_database_export test_genome_database_export)
    target_link_libraries(test_genome_database_export exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(test_checkpoint test_checkpoint)
    target_link_libraries(test_checkpoint exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(one_layer one_layer)
    target_link_libraries(one_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(two_layer two_layer)
    target_link_libraries(two_layer exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(lenet lenet)
    target_link_libraries(lenet exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(lenet_no_pool lenet_no_pool)
    target_link_libraries(lenet_no_pool exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(large_image_lenet_new large_image_lenet_new)
    target_link_libraries(large_image_lenet_new exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(large_image_lenet large_image_lenet)
    target_link_libraries(large_image_lenet exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(export_genome export_genome)
    target_link_libraries(export_genome exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)

    add_executable(generate_gv generate_gv)
    target_link_libraries(generate_gv exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES} pthread)
ELSE (MYSQL_FOUND)
    add_executable(evaluate_population evaluate_population)
    target_link_libraries(evaluate_population exact_strategy exact_common exact_image_tools pthread)

    add_executable(check_gradients check_gradients)
    target_link_libraries(check_gradients exact_strategy exact_common exact_image_tools pthread)

    add_executable(retrain_cnn retrain_cnn)
    target_link_libraries(retrain_cnn exact_strategy exact_common exact_image_tools pthread)

    add_executable(evaluate_large_image_cnn evaluate_large_image_cnn)
    target_link_libraries(evaluate_large_image_cnn exact_strategy exact_common exact_image_tools pthread)

    add_executable(evaluate_cnn evaluate_cnn)
    target_link_libraries(evaluate_cnn exact_strategy exact_common exact_image_tools pthread)

    add_executable(test_checkpoint test_checkpoint)
    target_link_libraries(test_checkpoint exact_strategy exact_common exact_image_tools pthread)

    add_executable(one_layer one_layer)
    target_link_libraries(one_layer exact_strategy exact_common exact_image_tools pthread)

    add_executable(two_layer two_layer)
    target_link_libraries(two_layer exact_strategy exact_common exact_image_tools pthread)

    add_executable(lenet lenet)
    target_link_libraries(lenet exact_strategy exact_common exact_image_tools pthread)

    add_executable(lenet_no_pool lenet_no_pool)
    target_link_libraries(lenet_no_pool exact_strategy exact_common exact_image_tools pthread)

    add_executable(large_image_lenet_new large_image_lenet_new)
    target_link_libraries(large_image_lenet_new exact_strategy exact_common exact_image_tools pthread)

    add_executable(large_image_lenet large_image_lenet)
    target_link_libraries(large_image_lenet exact_strategy exact_common exact_image_tools pthread)
ENDIF (MYSQL_FOUND)
