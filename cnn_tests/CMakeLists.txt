add_executable(evaluate_population evaluate_population.cxx)
target_link_libraries(evaluate_population exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(check_gradients check_gradients.cxx)
target_link_libraries(check_gradients exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(retrain_cnn retrain_cnn.cxx)
target_link_libraries(retrain_cnn exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

add_executable(test_checkpoint test_checkpoint.cxx)
target_link_libraries(test_checkpoint exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

if (MYSQL_FOUND)
    add_executable(export_genome export_genome.cxx)
    target_link_libraries(export_genome exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

    add_executable(test_genome_database_export test_genome_database_export.cxx)
    target_link_libraries(test_genome_database_export exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)
endif (MYSQL_FOUND)

add_executable(generate_gv generate_gv.cxx)
target_link_libraries(generate_gv exact_strategy exact_common exact_image_tools ${MYSQL_LIBRARIES}  ${TIFF_LIBRARIES} pthread)

